#!/usr/bin/env python3
"""Collect RAGAS samples from the real Session 11 generation pipeline.

For each query in ``evals/golden_generation_s11.json`` this runs the REAL RAG
generation pipeline (brief -> embed -> retrieve -> assemble -> generate) and
records the four inputs RAGAS needs:

    question      -> the estimation request (golden ``query``)
    answer        -> the estimate the pipeline generated, rendered as text
    contexts      -> the retrieved chunks placed in the context block
    ground_truth  -> the expert reference estimate (golden ``ground_truth``)

The script only COLLECTS samples. Scoring happens in the isolated RAGAS venv
with ``scripts/score_ragas_s11.py`` (see the spec, Paso 0.1), because the
project's langchain stack and RAGAS's pinned langchain stack conflict.

Prerequisites (host run):
    * estimator/.env with OPEN_AI_KEY
    * the stack up and the budget corpus ingested
      (uv run python scripts/query_examples.py)
    * the dev environment synced: uv sync

Usage::

    uv run python scripts/eval_ragas_s11.py --collect-only /tmp/s11_samples.json
    uv run python scripts/eval_ragas_s11.py --limit 2 --collect-only /tmp/s11_samples.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.dependencies import (  # noqa: E402
    get_runtime_retrieval_config,
    get_semantic_retriever,
    get_token_encoder,
)
from app.generation.rag.brief_extractor import (  # noqa: E402
    compose_retrieval_text,
    extract_project_brief,
)
from app.generation.rag.context_builder import (  # noqa: E402
    TokenEncoder,
    assemble_context,
    fit_chunks_to_budget,
)
from app.generation.rag.grounded_estimator import generate_grounded_estimate  # noqa: E402
from app.generation.rag.grounding import verify_citations_for_chunks  # noqa: E402
from app.generation.rag.retriever import SemanticRetriever  # noqa: E402
from app.generation.rag.schemas import GroundedEstimate  # noqa: E402

GOLDEN_PATH = ROOT / "evals" / "golden_generation_s11.json"

# Permissive threshold: we want the generator to see context for every query so
# RAGAS grades GENERATION, not the soft-fail gate.
GENERATION_THRESHOLD = 1.2


def render_estimate_text(estimate: GroundedEstimate) -> str:
    """Render a GroundedEstimate as the plain-text 'answer' RAGAS scores."""
    if estimate.confidence == "insufficient":
        return (
            "Insufficient context to produce an estimate. "
            f"{estimate.insufficient_context_explanation or ''}"
        ).strip()

    lines: list[str] = []
    if estimate.total_engineer_days is not None:
        lines.append(f"Total: {estimate.total_engineer_days} engineer-days.")
    if estimate.duration_weeks is not None:
        lines.append(f"Duration: {estimate.duration_weeks} weeks.")
    lines.append(f"Confidence: {estimate.confidence}.")
    for module in estimate.modules:
        lines.append(f"Module {module.name}:")
        for task in module.tasks:
            days = "n/a" if task.engineer_days is None else f"{task.engineer_days}d"
            tag = "grounded" if task.grounded else "no-data"
            citations = ", ".join(
                f"{reference.chunk_id}->{reference.document_id}" for reference in task.citations
            )
            suffix = f" sources=[{citations}]" if citations else ""
            lines.append(f"  - {task.name}: {days} [{tag}]{suffix}")
    if estimate.assumptions:
        lines.append(
            "Assumptions: " + "; ".join(assumption.description for assumption in estimate.assumptions)
        )
    lines.append(f"Reasoning: {estimate.reasoning}")
    return "\n".join(lines)


async def collect_sample(
    item: dict[str, Any],
    retriever: SemanticRetriever,
    encoder: TokenEncoder,
) -> dict[str, Any]:
    """Run the pipeline for one golden item and return a RAGAS-ready sample."""
    transcript = item.get("transcript") or item["query"]
    brief = await extract_project_brief(transcript)
    retrieval_text = compose_retrieval_text(brief)

    runtime = get_runtime_retrieval_config()
    retrieval = await retriever.retrieve_chunks(
        query_text=retrieval_text,
        search_mode=runtime.effective_search_mode(),
        rerank=runtime.effective_rerank(),
        top_k=settings.rag_retrieval_top_k,
        recall_k=settings.retrieval_recall_top_k,
        rerank_top_n=settings.rerank_top_n,
        distance_threshold=GENERATION_THRESHOLD,
        rrf_k=settings.rrf_k,
        chunk_types=["budget_component"],
    )

    kept = fit_chunks_to_budget(
        retrieval.chunks,
        settings.rag_max_context_tokens,
        encoder,
    )
    context_block = assemble_context(kept)
    estimate = await generate_grounded_estimate(context_block, brief)
    report = verify_citations_for_chunks(estimate, kept)

    return {
        "id": item["id"],
        "question": item["query"],
        "answer": render_estimate_text(estimate),
        "contexts": [chunk.content for chunk in kept],
        "ground_truth": item["ground_truth"],
        "citation_report": report.model_dump(),
    }


async def _run(args: argparse.Namespace) -> int:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    retriever = get_semantic_retriever()
    if retriever is None:
        raise SystemExit("Embedding service is not configured (OPEN_AI_KEY?).")

    encoder = get_token_encoder()
    items = golden["queries"]
    if args.limit is not None:
        items = items[: args.limit]

    samples: list[dict[str, Any]] = []
    for item in items:
        print(f"Collecting {item['id']} ...")
        samples.append(await collect_sample(item, retriever, encoder))

    payload = {
        "judge_model": golden.get("judge_model", "gpt-4o-mini"),
        "embedding_model": golden.get("embedding_model", "text-embedding-3-small"),
        "samples": samples,
    }
    destination = args.collect_only
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {destination}")
    print(
        "Next: score in the isolated RAGAS venv:\n"
        f"  .venv-ragas/bin/python scripts/score_ragas_s11.py {destination} "
        "--out evals/ragas_baseline_s11.json"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collect-only",
        type=Path,
        required=True,
        metavar="PATH",
        help="Write the collected samples to PATH (JSON).",
    )
    parser.add_argument("--limit", type=int, default=None)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
