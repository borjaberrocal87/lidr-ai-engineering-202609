#!/usr/bin/env python3
"""Mide la latencia SQL de las consultas de referencia.

El objetivo es aislar lo que el índice puede mejorar: el embedding de la query
se paga una vez fuera del cronómetro y cada medición ejecuta únicamente el
``SELECT`` de k-NN. Sirve como baseline antes de crear HNSW y como después para
exponer la diferencia real.

Uso::

    docker compose exec api python -m scripts.rag_ops.baseline_latency
    docker compose exec api python -m scripts.rag_ops.baseline_latency --runs 5
"""

from __future__ import annotations

import argparse

from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding_pipeline.store.repository import ChunkStore
from app.persistence.database import get_async_session_factory
from scripts.rag_ops.common import (
    DEFAULT_K,
    Stopwatch,
    embed_queries,
    format_table,
    latency_stats,
    latency_summary,
    print_footer,
    print_header,
    require_embedder,
    run_async,
    truncate,
    vector_literal,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latencia SQL pura (sin embedding) de las consultas de referencia.",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Chunks devueltos por query.")
    parser.add_argument("--runs", type=int, default=3, help="Mediciones por query.")
    parser.add_argument("--warmup", type=int, default=1, help="Ejecuciones de calentamiento.")
    args = parser.parse_args()
    if args.k < 1 or args.runs < 1 or args.warmup < 0:
        parser.error("--k y --runs deben ser >= 1, y --warmup >= 0")
    return args


async def _measure_query(
    session: AsyncSession,
    store: ChunkStore,
    vector: list[float],
    *,
    k: int,
    runs: int,
    warmup: int,
) -> list[float]:
    for _ in range(warmup):
        await store.search(session, query_vector=vector, k=k)

    samples: list[float] = []
    for _ in range(runs):
        with Stopwatch() as watch:
            await store.search(session, query_vector=vector, k=k)
        samples.append(watch.elapsed_ms)
    return samples


async def _main(args: argparse.Namespace) -> None:
    print_header("Latencia SQL del retrieval")
    embedder = require_embedder()
    queries = embed_queries(embedder)

    store = ChunkStore()
    session_factory = get_async_session_factory()
    rows: list[list[str]] = []
    all_samples: list[float] = []

    async with session_factory() as session:
        for index, (label, _query, vector) in enumerate(queries, start=1):
            samples = await _measure_query(
                session,
                store,
                vector,
                k=args.k,
                runs=args.runs,
                warmup=args.warmup,
            )
            all_samples.extend(samples)
            mean, deviation = latency_stats(samples)
            rows.append(
                [
                    str(index),
                    truncate(label, 44),
                    f"{mean:.1f}",
                    f"{deviation:.1f}",
                    f"{min(samples):.1f}",
                    f"{max(samples):.1f}",
                ]
            )

    print()
    print(
        format_table(
            ["#", "consulta", "media ms", "desv. ms", "mín. ms", "máx. ms"],
            rows,
        )
    )
    print_footer(
        f"Modo: {store.search_mode} · k={args.k} · "
        f"{args.runs} mediciones/query · {latency_summary(all_samples)}"
    )

    first_vector = queries[0][2]
    print(
        "\nLiteral pgvector de la primera query (para pegar en psql):\n"
        f"{truncate(vector_literal(first_vector), 180)}"
    )


if __name__ == "__main__":
    run_async(_main(_parse_args()))
