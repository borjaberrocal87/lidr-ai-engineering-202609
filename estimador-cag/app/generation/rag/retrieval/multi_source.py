"""Pipeline de recuperación multifuente (Sesión 10).

Ensambla todas las técnicas en el orden que impone el coste — "lo barato y
excluyente al principio; lo caro y fino al final; lo blando al cierre":

    plan de consulta → enrutado → filtros duros → búsqueda híbrida → fusión
                     → reranking → vigencia → top-k

Cada etapa está gobernada por :class:`StageConfig`, así que el pipeline es el
camino MÁXIMO, no el obligatorio, y cada técnica se puede medir en aislamiento.
El endpoint construye un ``StageConfig`` desde settings/runtime; el arnés de
evaluación construye configuraciones nombradas como DATOS.

La fusión se diferencia por técnica: las variantes de expansión se fusionan por
consenso (RRF) y las sub-consultas de descomposición por cobertura
(round-robin). Entre fuentes la fusión también es round-robin, con clave
``(collection, id)`` porque los ids de chunk solo son únicos DENTRO de una
fuente. Cada chunk conserva su etiqueta de procedencia.

Las dependencias (embedder, store, sesión, wrapper LLM y reranker) se inyectan:
este módulo no importa el composition root y respeta ``test_layering.py``.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import structlog

from app.config import Settings
from app.generation.rag.errors import RagRetrievalError
from app.generation.rag.retrieval.freshness import apply_freshness
from app.generation.rag.retrieval.fusion import reciprocal_rank_fusion, round_robin_merge
from app.generation.rag.retrieval.pipeline import CollectionHits, hybrid_search_one
from app.generation.rag.retrieval.query_plan import (
    DECOMPOSE,
    SubQuery,
    transform_query,
)
from app.generation.rag.retrieval.routing import RoutingDecision, route
from app.generation.rag.retrieval.sources import (
    Collection,
    HardFilters,
    spec_for,
)
from app.generation.rag.schemas import RetrievedChunk

log = structlog.get_logger()


@dataclass(frozen=True)
class StageConfig:
    """Qué etapas corren y con qué parámetros. Datos puros, sin I/O."""

    routing_enabled: bool = True
    query_plan_enabled: bool = True
    search_mode: str = "hybrid"
    rerank: bool = True
    freshness_enabled: bool = False
    top_k: int = 5
    recall_k: int = 50
    distance_threshold: float = 2.0
    rrf_k: int = 60
    half_life_days: int = 900
    max_subqueries: int = 4
    max_targets: int = 3
    router_model: str | None = None
    query_plan_model: str | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        search_mode: str,
        rerank: bool,
        routing_enabled: bool | None = None,
        query_plan_enabled: bool | None = None,
        freshness_enabled: bool | None = None,
        top_k: int | None = None,
        distance_threshold: float | None = None,
    ) -> StageConfig:
        """Construye la configuración efectiva desde settings, con overrides."""
        return cls(
            routing_enabled=(
                settings.retrieval_routing_enabled if routing_enabled is None else routing_enabled
            ),
            query_plan_enabled=(
                settings.retrieval_query_plan_enabled
                if query_plan_enabled is None
                else query_plan_enabled
            ),
            search_mode=search_mode,
            rerank=rerank,
            freshness_enabled=(
                settings.retrieval_freshness_enabled
                if freshness_enabled is None
                else freshness_enabled
            ),
            top_k=top_k or settings.rerank_top_n,
            recall_k=settings.retrieval_recall_top_k,
            distance_threshold=(
                settings.rag_retrieval_distance_threshold
                if distance_threshold is None
                else distance_threshold
            ),
            rrf_k=settings.rrf_k,
            half_life_days=settings.freshness_half_life_days,
            max_subqueries=settings.query_max_subqueries,
            max_targets=settings.router_max_targets,
            router_model=settings.router_model,
            query_plan_model=settings.query_plan_model,
        )


@dataclass
class MultiSourceOutcome:
    """Todo lo que expone el endpoint: resultados + cómo se obtuvieron."""

    chunks: list[RetrievedChunk]
    routing: RoutingDecision
    technique: str
    subqueries: list[SubQuery]
    cardinality: dict[str, int] = field(default_factory=dict)
    low_confidence: bool = True


def _chunk_key(chunk: RetrievedChunk) -> tuple[str, int]:
    """Identidad de deduplicación entre fuentes (los ids solo únicos por tabla)."""
    return (chunk.collection, chunk.id)


def _sigmoid(value: float) -> float:
    """Mapea un logit del cross-encoder a ``(0, 1)`` no negativo.

    La vigencia es un multiplicador 0..1; sin esto, un logit negativo se acercaría
    a cero y el orden se invertiría.
    """
    return 1.0 / (1.0 + math.exp(-value))


def _fuse_collection(
    hits_per_subquery: list[CollectionHits],
    *,
    technique: str,
    search_mode: str,
    rrf_k: int,
) -> list[RetrievedChunk]:
    """Fusiona los resultados de una fuente para todas las sub-consultas.

    Expansión/directo → RRF (consenso entre reformulaciones); descomposición →
    round-robin (cobertura entre temas).
    """
    candidates: dict[int, RetrievedChunk] = {}
    for hits in hits_per_subquery:
        candidates.update(hits.candidates)

    ordered_per_subquery = [
        hits.fused_order(search_mode=search_mode, rrf_k=rrf_k) for hits in hits_per_subquery
    ]

    if technique == DECOMPOSE:
        return round_robin_merge(ordered_per_subquery, key=lambda chunk: chunk.id)

    fused = reciprocal_rank_fusion(
        [[chunk.id for chunk in ordered] for ordered in ordered_per_subquery], k=rrf_k
    )
    return [candidates[cid] for cid, _score in fused]


async def retrieve_multi_source(
    *,
    query_text: str,
    embedder: Any,
    store: Any,
    session_factory: Any,
    stages: StageConfig,
    reference_date: date,
    llm_wrapper: Any = None,
    reranker: Any = None,
    explicit_collections: list[Collection] | None = None,
    hard_filters: HardFilters | None = None,
) -> MultiSourceOutcome:
    """Ejecuta el pipeline multifuente y devuelve resultados + diagnóstico."""
    hard_filters = hard_filters or HardFilters()

    # Etapas 1+2 — plan de consulta y enrutado (independientes, en paralelo).
    plan, routing = await asyncio.gather(
        transform_query(
            query_text,
            enabled=stages.query_plan_enabled,
            llm_wrapper=llm_wrapper,
            model=stages.query_plan_model,
            max_subqueries=stages.max_subqueries,
        ),
        route(
            query_text,
            explicit=explicit_collections,
            rules_enabled=stages.routing_enabled,
            classifier_enabled=stages.routing_enabled,
            max_targets=stages.max_targets,
            model=stages.router_model,
            llm_wrapper=llm_wrapper,
        ),
    )

    # Embebe cada sub-consulta una vez (en paralelo).
    try:
        embeddings = await asyncio.gather(
            *(asyncio.to_thread(embedder.embed_one, sq.query) for sq in plan.subqueries)
        )
    except Exception as exc:
        log.error(
            "multi_source_embed_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        raise RagRetrievalError("Failed to embed sub-queries.") from exc

    # Etapas 3+4 — filtros duros + búsqueda híbrida, por (fuente, sub-consulta),
    # cada una en su propia sesión para que el fan-out sea paralelo y seguro.
    async def _search(
        collection: Collection, subquery: SubQuery, embedding: list[float]
    ) -> CollectionHits:
        spec = spec_for(collection)
        extra_filters = spec.hard_filter_clauses(hard_filters)
        async with session_factory() as session:
            return await hybrid_search_one(
                session,
                store,
                collection=collection,
                query_embedding=embedding,
                query_text=subquery.query,
                search_mode=stages.search_mode,
                recall_k=stages.recall_k,
                top_k=stages.top_k,
                distance_threshold=stages.distance_threshold,
                will_rerank=stages.rerank,
                extra_filters=extra_filters,
            )

    try:
        flat = await asyncio.gather(
            *(
                _search(collection, subquery, embedding)
                for collection in routing.targets
                for subquery, embedding in zip(plan.subqueries, embeddings, strict=True)
            )
        )
    except Exception as exc:
        log.error(
            "multi_source_search_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        raise RagRetrievalError("Vector store query failed.") from exc

    # Reagrupa los resultados planos por fuente (las sub-consultas son contiguas).
    per_collection: dict[Collection, list[CollectionHits]] = {
        collection: [] for collection in routing.targets
    }
    index = 0
    for collection in routing.targets:
        for _subquery in plan.subqueries:
            per_collection[collection].append(flat[index])
            index += 1

    # Etapa 5 — fusión dentro de cada fuente y round-robin entre fuentes.
    cardinality: dict[str, int] = {}
    collection_orders: list[list[RetrievedChunk]] = []
    for collection, hits_list in per_collection.items():
        cardinality[collection.value] = hits_list[0].candidates_evaluated if hits_list else 0
        log.info(
            "multi_source_collection_cardinality",
            collection=collection.value,
            candidates_evaluated=cardinality[collection.value],
        )
        ordered = _fuse_collection(
            hits_list,
            technique=plan.technique,
            search_mode=stages.search_mode,
            rrf_k=stages.rrf_k,
        )
        if ordered:
            collection_orders.append(ordered)

    pool = round_robin_merge(collection_orders, key=_chunk_key)[: stages.recall_k]

    # Etapa 6 — reranking con scores, o score posicional si no hay reranker.
    if stages.rerank and pool and reranker is not None:
        reranked = await asyncio.to_thread(
            reranker.rerank_with_scores, query_text, pool, top_n=len(pool)
        )
        scored = [(chunk, _sigmoid(score)) for chunk, score in reranked]
    else:
        scored = [(chunk, 1.0 / (position + 1)) for position, chunk in enumerate(pool)]

    # Etapa 7 — vigencia (suave, al cierre) y corte final a top_k.
    if stages.freshness_enabled:
        scored = apply_freshness(
            scored,
            half_life_days=stages.half_life_days,
            reference_date=reference_date,
        )
    final_scored = scored[: stages.top_k]
    final = [chunk.model_copy(update={"relevance_score": score}) for chunk, score in final_scored]

    log.info(
        "multi_source_done",
        technique=plan.technique,
        routing_level=routing.level,
        targets=[t.value for t in routing.targets],
        subqueries=len(plan.subqueries),
        pool=len(pool),
        results=len(final),
    )
    return MultiSourceOutcome(
        chunks=final,
        routing=routing,
        technique=plan.technique,
        subqueries=plan.subqueries,
        cardinality=cardinality,
        low_confidence=not final,
    )
