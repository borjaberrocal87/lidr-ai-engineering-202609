#!/usr/bin/env python3
"""Compara el índice vectorial completo con el índice halfvec.

Ambos índices pueden convivir: la query completa ordena por ``embedding`` y la
de media precisión por ``(embedding::halfvec(1536))``. El script mide latencia,
calcula el solapamiento del top-k y verifica con ``EXPLAIN`` qué índice usa
realmente cada plan.

Requiere ``01_hnsw.sql`` y ``03_halfvec.sql`` ejecutados.

Uso::

    docker compose exec api python -m scripts.rag_ops.index_compare
"""

from __future__ import annotations

import argparse
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding_pipeline.store.models import EMBEDDING_DIMENSIONS
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
        description="Compara top-k y latencia entre índice vector y halfvec.",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k comparado.")
    parser.add_argument("--runs", type=int, default=3, help="Mediciones por query.")
    parser.add_argument("--warmup", type=int, default=1, help="Ejecuciones de calentamiento.")
    args = parser.parse_args()
    if args.k < 1 or args.runs < 1 or args.warmup < 0:
        parser.error("--k y --runs deben ser >= 1, y --warmup >= 0")
    return args


def _find_index_name(plan: dict[str, Any]) -> str:
    """Busca el nombre del índice en el árbol de un plan de Postgres."""
    index_name = plan.get("Index Name")
    if index_name:
        return str(index_name)
    for child in plan.get("Plans", []):
        found = _find_index_name(child)
        if found != "—":
            return found
    return "—"


async def _explain_index(
    session: AsyncSession,
    vector: list[float],
    *,
    halfvec: bool,
) -> str:
    expression = f"(embedding::halfvec({EMBEDDING_DIMENSIONS}))" if halfvec else "embedding"
    cast_type = f"halfvec({EMBEDDING_DIMENSIONS})" if halfvec else "vector"
    statement = text(
        "EXPLAIN (FORMAT JSON) "  # noqa: S608 — expresión y tipo son constantes del código
        "SELECT id FROM chunks "
        f"ORDER BY {expression} <=> CAST(:query AS {cast_type}) "
        "LIMIT 5"
    )
    result = await session.execute(statement, {"query": vector_literal(vector)})
    payload = result.scalar_one()
    plan = payload["Plan"] if isinstance(payload, dict) else payload[0]["Plan"]
    return _find_index_name(plan)


async def _measure(
    session: AsyncSession,
    store: ChunkStore,
    vector: list[float],
    *,
    k: int,
    runs: int,
    warmup: int,
) -> tuple[list[float], list[int]]:
    for _ in range(warmup):
        await store.search(session, query_vector=vector, k=k)

    samples: list[float] = []
    top_ids: list[int] = []
    for _ in range(runs):
        with Stopwatch() as watch:
            matches = await store.search(session, query_vector=vector, k=k)
        samples.append(watch.elapsed_ms)
        top_ids = [match.chunk_id for match in matches]
    return samples, top_ids


async def _main(args: argparse.Namespace) -> None:
    print_header("Índice vector completo vs halfvec")
    queries = embed_queries(require_embedder())
    full_store = ChunkStore(use_halfvec=False)
    half_store = ChunkStore(use_halfvec=True)
    session_factory = get_async_session_factory()

    rows: list[list[str]] = []
    all_full: list[float] = []
    all_half: list[float] = []

    async with session_factory() as session:
        first_vector = queries[0][2]
        full_index = await _explain_index(session, first_vector, halfvec=False)
        half_index = await _explain_index(session, first_vector, halfvec=True)
        print(f"Índice vectorial usado: {full_index}")
        print(f"Índice halfvec usado:   {half_index}\n")

        for index, (label, _query, vector) in enumerate(queries, start=1):
            full_samples, full_ids = await _measure(
                session,
                full_store,
                vector,
                k=args.k,
                runs=args.runs,
                warmup=args.warmup,
            )
            half_samples, half_ids = await _measure(
                session,
                half_store,
                vector,
                k=args.k,
                runs=args.runs,
                warmup=args.warmup,
            )
            all_full.extend(full_samples)
            all_half.extend(half_samples)

            overlap = len(set(full_ids) & set(half_ids)) / args.k
            full_mean, _ = latency_stats(full_samples)
            half_mean, _ = latency_stats(half_samples)
            rows.append(
                [
                    str(index),
                    truncate(label, 42),
                    f"{overlap:.2f}",
                    f"{full_mean:.1f}",
                    f"{half_mean:.1f}",
                    f"{half_mean - full_mean:+.1f}",
                ]
            )

    print(
        format_table(
            ["#", "consulta", "overlap", "vector ms", "halfvec ms", "Δ ms"],
            rows,
        )
    )
    full_mean, _ = latency_stats(all_full)
    half_mean, _ = latency_stats(all_half)
    print_footer(
        f"Vector: {latency_summary(all_full)}\n"
        f"Halfvec: {latency_summary(all_half)}\n"
        f"Media halfvec - vector: {half_mean - full_mean:+.1f} ms"
    )
    if full_index == "—" or half_index == "—":
        print(
            "Aviso: algún plan no usó índice. Revisa que 01_hnsw.sql y "
            "03_halfvec.sql se hayan ejecutado sobre este corpus."
        )


if __name__ == "__main__":
    run_async(_main(_parse_args()))
