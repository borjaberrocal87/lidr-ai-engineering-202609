#!/usr/bin/env python3
"""Barre ``hnsw.ef_search`` y mide el compromiso recall/latencia.

La verdad de fondo se calcula con un sequential scan forzado (exacta por
definición). Después, para cada valor de ``ef_search``, se ejecutan las mismas
consultas contra el índice y se compara el top-5 devuelto con el top-5 exacto.
El script marca con ★ el valor más pequeño que mantiene el recall dentro de
0,005 del mejor observado (o ≥ 0,99).

Requiere el índice HNSW creado (``scripts/rag_ops/sql/01_hnsw.sql``).

Uso::

    docker compose exec api python -m scripts.rag_ops.ef_search_sweep
"""

from __future__ import annotations

import argparse
import statistics

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.embedding_pipeline.store.repository import ChunkStore
from app.persistence.database import get_async_session_factory
from scripts.rag_ops.common import (
    DEFAULT_K,
    GROUND_TRUTH_K,
    Stopwatch,
    embed_queries,
    format_table,
    print_footer,
    print_header,
    require_embedder,
    run_async,
    truncate,
)

EF_SEARCH_VALUES = (10, 20, 40, 80, 120, 200)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Barrido de hnsw.ef_search con latencia y recall@5.",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k del recall.")
    parser.add_argument("--runs", type=int, default=3, help="Mediciones por query y valor.")
    parser.add_argument("--warmup", type=int, default=1, help="Ejecuciones de calentamiento.")
    args = parser.parse_args()
    if args.k < 1 or args.runs < 1 or args.warmup < 0:
        parser.error("--k y --runs deben ser >= 1, y --warmup >= 0")
    return args


async def _ground_truth(
    session: AsyncSession,
    store: ChunkStore,
    queries: list[tuple[str, str, list[float]]],
) -> dict[str, list[int]]:
    truth: dict[str, list[int]] = {}
    for label, _query, vector in queries:
        async with session.begin():
            await session.execute(text("SET LOCAL enable_indexscan = off"))
            matches = await store.search(session, query_vector=vector, k=GROUND_TRUTH_K)
        truth[label] = [match.chunk_id for match in matches]
    return truth


def _recall_at_k(reference: list[int], candidate: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(reference[:k]) & set(candidate[:k])) / k


def _p95(samples: list[float]) -> float:
    if len(samples) < 20:
        return max(samples)
    return statistics.quantiles(samples, n=20)[-1]


def _recommend_ef(results: list[tuple[int, float, float, float]]) -> int:
    """Elige el ef_search más pequeño que no sacrifica recall de forma visible."""
    best_recall = max(result[1] for result in results)
    target = 0.99 if best_recall >= 0.99 else best_recall - 0.005
    return next(
        (ef for ef, recall, _mean, _p95_value in results if recall >= target),
        results[-1][0],
    )


async def _main(args: argparse.Namespace) -> None:
    print_header("Barrido de ef_search")
    settings = get_settings()
    store = ChunkStore(use_halfvec=settings.search_use_halfvec)
    queries = embed_queries(require_embedder())
    session_factory = get_async_session_factory()

    results: list[tuple[int, float, float, float]] = []
    async with session_factory() as session:
        truth = await _ground_truth(session, store, queries)
        print(f"Verdad de fondo: top-{GROUND_TRUTH_K} por seq scan forzado.\n")

        for ef_search in EF_SEARCH_VALUES:
            latencies: list[float] = []
            recalls: list[float] = []
            async with session.begin():
                await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
                for label, _query, vector in queries:
                    for _ in range(args.warmup):
                        await store.search(session, query_vector=vector, k=args.k)

                    top_ids: list[int] = []
                    for _ in range(args.runs):
                        with Stopwatch() as watch:
                            matches = await store.search(
                                session,
                                query_vector=vector,
                                k=args.k,
                            )
                        latencies.append(watch.elapsed_ms)
                        top_ids = [match.chunk_id for match in matches]

                    recalls.append(_recall_at_k(truth[label], top_ids, min(args.k, GROUND_TRUTH_K)))

            mean_recall = statistics.fmean(recalls)
            mean_latency = statistics.fmean(latencies)
            results.append((ef_search, mean_recall, mean_latency, _p95(latencies)))

    best_recall = max(result[1] for result in results)
    recommended = _recommend_ef(results)

    rows = [
        [
            ("★ " if ef == recommended else "  ") + str(ef),
            f"{recall:.3f}",
            f"{mean_latency:.1f}",
            f"{p95_latency:.1f}",
        ]
        for ef, recall, mean_latency, p95_latency in results
    ]
    print(
        format_table(
            ["ef_search", "recall@k", "latencia media ms", "p95 ms"],
            rows,
        )
    )
    print_footer(
        f"Modo: {store.search_mode} · k={args.k} · "
        f"mejor recall={best_recall:.3f} · recomendado ef_search={recommended}"
    )
    print(
        "Recuerda: si no existe el índice HNSW, todos los valores miden el mismo "
        "sequential scan. Ejecuta antes scripts/rag_ops/sql/01_hnsw.sql."
    )
    print(f"Consultas evaluadas: {truncate(', '.join(q[0] for q in queries), 120)}")


if __name__ == "__main__":
    run_async(_main(_parse_args()))
