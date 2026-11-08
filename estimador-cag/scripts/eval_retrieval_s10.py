#!/usr/bin/env python3
"""Measure retrieval quality across the four Session 10 configurations.

Runs the golden set (``evals/golden_retrieval.json``) through the same
``retrieve()`` pipeline the API uses, for each of:

    A  Vector  / no rerank   (the Session 9 baseline)
    B  Hybrid  / no rerank
    C  Vector  / rerank
    D  Hybrid  / rerank

and reports precision@5 (mean over the queries) and query latency (mean over
measured runs) per configuration, plus a per-query precision breakdown.

Method notes:
* Queries are embedded ONCE upfront; embedding latency is an OpenAI round-trip
  shared by every config, so it is excluded from the timings (we measure the
  retrieval/rerank cost the techniques actually add).
* A permissive distance threshold is used so the top-5 is never truncated by the
  relevance floor — we are comparing RANKING quality, not the soft-fail gate.
* A retrieved chunk is relevant iff its parent ``budget_id`` is in the query's
  annotated ``relevant_budget_ids``. precision@5 = relevant_in_top5 / 5.

Usage (host, stack up + corpus ingested + OPEN_AI_KEY in .env)::

    uv run python scripts/eval_retrieval_s10.py

Reranking config runs download the cross-encoder weights on first use; verify
first with ``python -m app.generation.rag.retrieval.verify_reranker``.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.dependencies import (  # noqa: E402
    get_async_session_factory,
    get_chunk_store,
    get_reranker,
)
from app.generation.rag.retrieval.pipeline import retrieve  # noqa: E402
from app.generation.rag.schemas import RetrievedChunk  # noqa: E402
from scripts.rag_ops.common import require_embedder  # noqa: E402

GOLDEN_PATH = ROOT / "evals" / "golden_retrieval.json"

# (id, search label, rerank label, search_mode, rerank)
CONFIGS = [
    ("A", "Vector", "No", "vector", False),
    ("B", "Hybrid", "No", "hybrid", False),
    ("C", "Vector", "Yes", "vector", True),
    ("D", "Hybrid", "Yes", "hybrid", True),
]

# No effective relevance floor: we want a full top-k to grade ranking quality.
NO_FLOOR_THRESHOLD = 2.0
# Latency sampling per (query, config): one discarded warm-up + N measured runs.
MEASURED_RUNS = 3


def precision_at_k(chunks: list[RetrievedChunk], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k results whose parent budget is genuinely relevant."""
    top = chunks[:k]
    hits = sum(1 for chunk in top if chunk.budget_id in relevant_ids)
    return hits / k


async def _run_once(
    *,
    store: Any,
    session_factory: Any,
    query_embedding: list[float],
    query_text: str,
    search_mode: str,
    rerank: bool,
    settings: Any,
    chunk_types: list[str] | None,
    k: int,
    reranker: Any,
) -> Any:
    return await retrieve(
        store=store,
        session_factory=session_factory,
        query_embedding=query_embedding,
        query_text=query_text,
        search_mode=search_mode,
        rerank=rerank,
        top_k=k,
        recall_k=settings.retrieval_recall_top_k,
        rerank_top_n=k,
        distance_threshold=NO_FLOOR_THRESHOLD,
        rrf_k=settings.rrf_k,
        chunk_types=chunk_types,
        reranker=reranker,
    )


async def main() -> int:
    settings = get_settings()
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    queries = golden["queries"]
    chunk_types = golden.get("chunk_types")
    k = int(golden.get("k", 5))

    embedder = require_embedder()
    print(f"Embedding {len(queries)} golden queries (excluded from timings)...")
    embeddings = await asyncio.to_thread(
        lambda: {q["id"]: embedder.embed_one(q["query"]) for q in queries}
    )

    reranker = get_reranker()
    print("Warming up the cross-encoder (first load downloads weights)...")
    await asyncio.to_thread(reranker.load)

    store = get_chunk_store()
    session_factory = get_async_session_factory()

    results: dict[str, dict[str, Any]] = {
        cfg[0]: {"precisions": [], "latencies_ms": [], "per_query": {}} for cfg in CONFIGS
    }
    empty_warning = False

    for cfg_id, _search_label, _rerank_label, search_mode, rerank in CONFIGS:
        for query in queries:
            relevant = set(query["relevant_budget_ids"])
            embedding = embeddings[query["id"]]

            # Warm-up (discarded) then measured runs.
            await _run_once(
                store=store,
                session_factory=session_factory,
                query_embedding=embedding,
                query_text=query["query"],
                search_mode=search_mode,
                rerank=rerank,
                settings=settings,
                chunk_types=chunk_types,
                k=k,
                reranker=reranker,
            )
            samples: list[float] = []
            last = None
            for _ in range(MEASURED_RUNS):
                started = time.perf_counter()
                last = await _run_once(
                    store=store,
                    session_factory=session_factory,
                    query_embedding=embedding,
                    query_text=query["query"],
                    search_mode=search_mode,
                    rerank=rerank,
                    settings=settings,
                    chunk_types=chunk_types,
                    k=k,
                    reranker=reranker,
                )
                samples.append((time.perf_counter() - started) * 1000)

            if not last.chunks:
                empty_warning = True
            precision = precision_at_k(last.chunks, relevant, k)
            results[cfg_id]["precisions"].append(precision)
            results[cfg_id]["latencies_ms"].extend(samples)
            results[cfg_id]["per_query"][query["id"]] = precision

    _print_report(results, queries, k)
    if empty_warning:
        print(
            "\nWARNING: some configurations returned 0 chunks. Is the base corpus "
            "ingested? Run `docker compose run --rm api python scripts/query_examples.py`.",
            file=sys.stderr,
        )
    return 0


def _print_report(
    results: dict[str, dict[str, Any]],
    queries: list[dict[str, Any]],
    k: int,
) -> None:
    print(f"\n## Retrieval evaluation — precision@{k} and latency\n")
    print(f"| Config | Search | Reranking | Precision@{k} | Latency (ms) |")
    print("| --- | --- | --- | --- | --- |")
    for cfg_id, search_label, rerank_label, _mode, _rerank in CONFIGS:
        bucket = results[cfg_id]
        mean_precision = statistics.fmean(bucket["precisions"])
        mean_latency = statistics.fmean(bucket["latencies_ms"])
        print(
            f"| {cfg_id} | {search_label} | {rerank_label} | "
            f"{mean_precision:.2f} | {mean_latency:.1f} |"
        )

    print(f"\n### Per-query precision@{k}\n")
    header = "| Query | " + " | ".join(cfg[0] for cfg in CONFIGS) + " |"
    print(header)
    print("| --- | " + " | ".join("---" for _ in CONFIGS) + " |")
    for query in queries:
        row = [query["id"]] + [
            f"{results[cfg[0]]['per_query'][query['id']]:.2f}" for cfg in CONFIGS
        ]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
