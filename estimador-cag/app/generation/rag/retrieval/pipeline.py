"""Pipeline recall-then-rerank (Sesión 10).

``retrieve()`` sigue siendo el punto de entrada de una sola fuente y compone las
cuatro configuraciones que mide la sesión, detrás de dos interruptores que el
llamante resuelve (petición → runtime → settings):

* ``search_mode="vector"`` — solo k-NN denso (comportamiento de la Sesión 9).
* ``search_mode="hybrid"`` — ramas densa y léxica fusionadas con RRF.
* ``rerank=False`` — conserva el top ``top_k`` del orden de recall.
* ``rerank=True`` — recall ANCHO (``recall_k``, p. ej. 50) y luego el
  cross-encoder reordena y conserva el top ``rerank_top_n``. La etapa de recall
  solo tiene que no perder el documento relevante; el reranker lo sube.

Devuelve el mismo :class:`RetrievalOutcome` de siempre, de modo que todos los
consumidores existentes (orquestador, endpoints por etapa) no se enteran.

El primitivo de una sola fuente ``hybrid_search_one()`` se factoriza para que el
pipeline multifuente (enrutado, expansión, varias fuentes) reutilice exactamente
la misma lógica de recall y fusión.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.generation.rag.retrieval.fusion import reciprocal_rank_fusion
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.retrieval.sources import Collection, spec_for
from app.generation.rag.schemas import RetrievalOutcome, RetrievedChunk
from app.generation.rag.store.repository import (
    LEXICAL_ONLY_DISTANCE,
    ChunkMatch,
    ChunkStore,
)

log = structlog.get_logger()


class CollectionHits:
    """Recall de UNA (fuente, consulta): candidatos y rankings por rama.

    El pool mapea ``chunk_id → RetrievedChunk`` (los ids son únicos DENTRO de una
    fuente). ``vector_ids``/``lexical_ids`` son los órdenes por rama que el
    llamante fusiona: RRF para variantes de expansión, round-robin para
    descomposición.
    """

    def __init__(
        self,
        *,
        collection: Collection,
        candidates: dict[int, RetrievedChunk],
        vector_ids: list[int],
        lexical_ids: list[int],
        candidates_evaluated: int,
    ) -> None:
        self.collection = collection
        self.candidates = candidates
        self.vector_ids = vector_ids
        self.lexical_ids = lexical_ids
        self.candidates_evaluated = candidates_evaluated

    def fused_order(self, *, search_mode: str, rrf_k: int) -> list[RetrievedChunk]:
        """Chunks de mejor a peor: RRF de ambas ramas si hay híbrida."""
        if search_mode == "hybrid":
            fused = reciprocal_rank_fusion([self.vector_ids, self.lexical_ids], k=rrf_k)
            return [self.candidates[cid] for cid, _score in fused]
        return [self.candidates[cid] for cid in self.vector_ids]


async def hybrid_search_one(
    session: AsyncSession,
    store: ChunkStore,
    *,
    collection: Collection,
    query_embedding: list[float],
    query_text: str,
    search_mode: str,
    recall_k: int,
    top_k: int,
    distance_threshold: float,
    will_rerank: bool,
    sectors: list[str] | None = None,
    project_year_min: int | None = None,
    project_year_max: int | None = None,
    chunk_types: list[str] | None = None,
    extra_filters: list | None = None,
) -> CollectionHits:
    """Ejecuta las ramas vectorial (e híbrida) sobre UNA fuente.

    El recall es ancho siempre que una etapa posterior (fusión o rerank) vaya a
    reordenar; solo se recorta a ``top_k`` en el camino vectorial puro, donde el
    orden ya es definitivo.
    """
    spec = spec_for(collection)
    wide = will_rerank or search_mode == "hybrid"
    vector_limit = recall_k if wide else top_k

    vector_matches, candidates_evaluated = await store.search_with_filters(
        session,
        query_vector=query_embedding,
        top_k=vector_limit,
        distance_threshold=distance_threshold,
        sectors=sectors,
        project_year_min=project_year_min,
        project_year_max=project_year_max,
        chunk_types=chunk_types,
        model=spec.model,
        extra_filters=extra_filters,
    )

    lexical_matches: list[ChunkMatch] = []
    if search_mode == "hybrid":
        lexical_matches = await store.search_lexical(
            session,
            query_text=query_text,
            top_k=recall_k,
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
            model=spec.model,
            extra_filters=extra_filters,
        )

    candidates: dict[int, RetrievedChunk] = {}
    for match in vector_matches:
        candidates[match.chunk_id] = spec.to_chunk(match, distance=match.distance)
    for match in lexical_matches:
        candidates.setdefault(
            match.chunk_id,
            spec.to_chunk(match, distance=LEXICAL_ONLY_DISTANCE),
        )

    return CollectionHits(
        collection=collection,
        candidates=candidates,
        vector_ids=[match.chunk_id for match in vector_matches],
        lexical_ids=[match.chunk_id for match in lexical_matches],
        candidates_evaluated=candidates_evaluated,
    )


async def retrieve(
    *,
    store: ChunkStore,
    session_factory: async_sessionmaker[AsyncSession],
    query_embedding: list[float],
    query_text: str,
    search_mode: str = "vector",
    rerank: bool = False,
    top_k: int = 10,
    recall_k: int = 50,
    rerank_top_n: int = 5,
    distance_threshold: float = 0.6,
    rrf_k: int = 60,
    collection: Collection = Collection.BUDGET,
    sectors: list[str] | None = None,
    project_year_min: int | None = None,
    project_year_max: int | None = None,
    chunk_types: list[str] | None = None,
    extra_filters: list | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> RetrievalOutcome:
    """Ejecuta retrieval vectorial/híbrido con reranking opcional.

    ``query_text`` es obligatorio para la rama léxica y para el reranker incluso
    cuando ``search_mode`` es ``"vector"`` (el reranker puntúa contra la consulta
    cruda). ``reranker`` se inyecta; cuando ``rerank`` es True debe estar
    presente. Devuelve el top de resultados de mejor a peor (orden del
    cross-encoder al rerankear; si no, orden de distancia/RRF).
    """
    if search_mode not in {"vector", "hybrid"}:
        raise ValueError(f"Unsupported search_mode: {search_mode!r}")
    if rerank and reranker is None:
        raise ValueError("rerank=True requires an injected reranker")

    started = time.perf_counter()

    async with session_factory() as session:
        hits = await hybrid_search_one(
            session,
            store,
            collection=collection,
            query_embedding=query_embedding,
            query_text=query_text,
            search_mode=search_mode,
            recall_k=recall_k,
            top_k=top_k,
            distance_threshold=distance_threshold,
            will_rerank=rerank,
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
            extra_filters=extra_filters,
        )

    ordered = hits.fused_order(search_mode=search_mode, rrf_k=rrf_k)

    if rerank and ordered and reranker is not None:
        final = await asyncio.to_thread(
            reranker.rerank,
            query_text,
            ordered[:recall_k],
            top_n=rerank_top_n,
        )
    else:
        final = ordered[:top_k]

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "rag.retrieval.end",
        query=query_text[:80],
        collection=collection.value,
        search_mode=search_mode,
        rerank=rerank,
        vector_hits=len(hits.vector_ids),
        lexical_hits=len(hits.lexical_ids),
        results=len(final),
        candidates_evaluated=hits.candidates_evaluated,
        store_search_mode=store.search_mode,
        search_time_ms=elapsed_ms,
    )
    return RetrievalOutcome(
        chunks=final,
        low_confidence=not final,
        candidates_evaluated=hits.candidates_evaluated,
    )
