"""Consenso de horas por tarea mediante búsqueda vectorial (Sesión 10).

La generación estructura-primero produce módulos → tareas SIN horas. Este módulo
deriva las horas de cada tarea desde el corpus histórico de tareas
(``chunk_type='historical_task'``, generado por ``scripts/build_task_corpus.py``):

1. compone un texto de búsqueda con la tarea (módulo + nombre + descripción),
2. lo embebe y ejecuta un k-NN filtrado por metadata sobre las tareas históricas,
3. combina las ``estimated_hours`` de los vecinos más cercanos en un único número
   mediante un CONSENSO PONDERADO POR DISTANCIA, con una fiabilidad que mezcla
   cercanía y acuerdo.

Una tarea cuyo vecino más cercano queda más lejos que el umbral recibe NO horas
(``has_match=False``), que la UI muestra en rojo para que la persona aporte el
número. El retriever no agrega nada: devuelve chunks; el consenso vive aquí como
función pura y testeable.

Las dependencias se inyectan (embedder, store, sesión, reranker); este módulo no
importa el composition root.
"""

from __future__ import annotations

import asyncio
import statistics

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.retrieval.pipeline import retrieve
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.retrieval.sources import Collection
from app.generation.rag.schemas import (
    RetrievedChunk,
    TaskHoursEstimate,
    TaskHoursModuleInput,
    TaskHoursResult,
    TaskNeighbor,
)
from app.generation.rag.store.repository import ChunkStore

log = structlog.get_logger()

_HISTORICAL_TASK_CHUNK_TYPE = "historical_task"
# Suelo del peso por distancia inversa: un match exacto (distancia ≈ 0) no debe
# explotar a infinito y silenciar al resto de vecinos.
_WEIGHT_EPS = 1e-3


def compose_task_search_text(module: str, name: str, description: str | None) -> str:
    """Compone el texto embebible de una tarea.

    Imita la forma del chunk histórico (módulo + nombre + descripción) para que
    consulta y corpus vivan en el mismo espacio de vocabulario.
    """
    parts = []
    if module:
        parts.append(f"Module: {module}")
    parts.append(f"Task: {name}")
    if description:
        parts.append(description)
    return "\n".join(parts)


def _consensus(neighbors: list[tuple[int, float]]) -> tuple[int, float, float]:
    """Consenso ponderado por distancia sobre vecinos ``(horas, distancia)``.

    Devuelve ``(horas, fiabilidad, dispersión)``:

    * ``horas`` — media ponderada por distancia inversa, redondeada.
    * ``fiabilidad`` — ``similitud_ponderada * (1 - min(dispersión, 1))`` en
      ``[0, 1]``: alta solo si los vecinos son CERCA y están de ACUERDO.
    * ``dispersión`` — coeficiente de variación de las horas (std/media), un
      número transparente de "cuánto discrepan".

    Precondición: ``neighbors`` no está vacío (el llamante trata el no-match).
    """
    weights = [1.0 / (_WEIGHT_EPS + distance) for _hours, distance in neighbors]
    total_w = sum(weights)
    hours_values = [hours for hours, _distance in neighbors]

    weighted_hours = sum(w * h for w, (h, _d) in zip(weights, neighbors, strict=True)) / total_w

    weighted_similarity = (
        sum(
            w * max(0.0, 1.0 - distance)
            for w, (_h, distance) in zip(weights, neighbors, strict=True)
        )
        / total_w
    )

    mean_hours = statistics.fmean(hours_values)
    if len(hours_values) > 1 and mean_hours > 0:
        dispersion = statistics.pstdev(hours_values) / mean_hours
    else:
        dispersion = 0.0

    reliability = weighted_similarity * (1.0 - min(dispersion, 1.0))
    reliability = max(0.0, min(1.0, reliability))
    return round(weighted_hours), round(reliability, 3), round(dispersion, 3)


async def estimate_one(
    module: str,
    name: str,
    description: str | None,
    *,
    store: ChunkStore,
    session_factory: async_sessionmaker[AsyncSession],
    embedder: OpenAIEmbedder,
    reranker: CrossEncoderReranker | None = None,
    top_k: int = 5,
    distance_threshold: float = 0.45,
    search_mode: str = "vector",
    rerank: bool = False,
    recall_k: int = 50,
    rerank_top_n: int = 5,
    rrf_k: int = 60,
) -> TaskHoursEstimate:
    """Estima las horas de una sola tarea desde el corpus histórico.

    La búsqueda pasa por el pipeline completo (híbrida + reranking opcional),
    filtrada a chunks ``historical_task``, de modo que los vecinos que alimentan
    el consenso son los que el cross-encoder juzga más relevantes, no solo los
    más cercanos en coseno.
    """
    search_text = compose_task_search_text(module, name, description)
    embedding = await asyncio.to_thread(embedder.embed_one, search_text)
    result = await retrieve(
        store=store,
        session_factory=session_factory,
        query_embedding=embedding,
        query_text=search_text,
        search_mode=search_mode,
        rerank=rerank,
        top_k=top_k,
        recall_k=recall_k,
        rerank_top_n=rerank_top_n,
        distance_threshold=distance_threshold,
        rrf_k=rrf_k,
        collection=Collection.BUDGET,
        chunk_types=[_HISTORICAL_TASK_CHUNK_TYPE],
        reranker=reranker if rerank else None,
    )

    # El umbral de distancia es la regla de "sin análogo". En híbrida, los
    # matches léxicos llegan con LEXICAL_ONLY_DISTANCE (1.0): si no tienen un
    # análogo vectorial suficientemente cerca, NO deben convertirse en horas.
    usable: list[RetrievedChunk] = [
        chunk
        for chunk in result.chunks
        if chunk.estimated_hours is not None and chunk.distance <= distance_threshold
    ]
    if not usable:
        return TaskHoursEstimate(module=module, task=name, has_match=False)

    neighbors = [
        TaskNeighbor(
            source_id=chunk.id,
            budget_id=chunk.budget_id,
            estimated_hours=int(chunk.estimated_hours or 0),
            distance=chunk.distance,
        )
        for chunk in usable
    ]
    hours, reliability, dispersion = _consensus(
        [(neighbor.estimated_hours, neighbor.distance) for neighbor in neighbors]
    )
    return TaskHoursEstimate(
        module=module,
        task=name,
        estimated_hours=hours,
        reliability=reliability,
        has_match=True,
        dispersion=dispersion,
        neighbors=neighbors,
    )


async def estimate_all(
    modules: list[TaskHoursModuleInput],
    *,
    store: ChunkStore,
    session_factory: async_sessionmaker[AsyncSession],
    embedder: OpenAIEmbedder,
    reranker: CrossEncoderReranker | None = None,
    top_k: int = 5,
    distance_threshold: float = 0.45,
    search_mode: str = "vector",
    rerank: bool = False,
    recall_k: int = 50,
    rerank_top_n: int = 5,
    rrf_k: int = 60,
) -> TaskHoursResult:
    """Estima las horas de todas las tareas de todos los módulos, en paralelo.

    Una búsqueda vectorial por tarea (fan-out con ``asyncio.gather``); el orden
    de las estimaciones devueltas coincide con el orden de las tareas enviadas.
    """
    coros = [
        estimate_one(
            module.name,
            task.name,
            task.description,
            store=store,
            session_factory=session_factory,
            embedder=embedder,
            reranker=reranker,
            top_k=top_k,
            distance_threshold=distance_threshold,
            search_mode=search_mode,
            rerank=rerank,
            recall_k=recall_k,
            rerank_top_n=rerank_top_n,
            rrf_k=rrf_k,
        )
        for module in modules
        for task in module.tasks
    ]
    estimates = await asyncio.gather(*coros)
    matched = sum(1 for estimate in estimates if estimate.has_match)
    log.info(
        "task_hours_done",
        tasks=len(estimates),
        matched=matched,
        flagged=len(estimates) - matched,
        top_k=top_k,
        distance_threshold=distance_threshold,
        search_mode=search_mode,
        rerank=rerank,
    )
    return TaskHoursResult(tasks=list(estimates))
