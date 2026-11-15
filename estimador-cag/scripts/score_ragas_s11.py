#!/usr/bin/env python3
"""Standalone RAGAS scorer for pre-collected Session 11 samples.

This script imports ONLY ragas + langchain-openai + datasets — never the app —
so it runs in an isolated venv, decoupled from the project's newer langchain
stack (which conflicts with ragas's hard ``langchain_community`` Vertex import).
Collect the samples first with::

    uv run python scripts/eval_ragas_s11.py --collect-only /tmp/s11_samples.json

then score them here::

    .venv-ragas/bin/python scripts/score_ragas_s11.py /tmp/s11_samples.json \
        --out evals/ragas_baseline_s11.json

ragas 0.4.x unconditionally imports ``langchain_community.chat_models.vertexai``
at module load; recent ``langchain-community`` dropped that path. We never use
Vertex (the judge is OpenAI), so we register lightweight stub modules before
importing ragas to satisfy the import without pulling the Vertex SDK.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _install_vertex_shims() -> None:
    """Satisfy ragas's hard Vertex imports without the Vertex SDK (unused here)."""
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401

        return
    except Exception:  # noqa: S110 - optional dependency probe
        pass

    class _Unavailable:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - defensive
            raise RuntimeError("Vertex AI is not available in this scorer venv.")

    chat_mod = types.ModuleType("langchain_community.chat_models.vertexai")
    chat_mod.ChatVertexAI = _Unavailable
    sys.modules["langchain_community.chat_models.vertexai"] = chat_mod

    try:
        import langchain_community.llms as llms_mod  # type: ignore

        if not hasattr(llms_mod, "VertexAI"):
            llms_mod.VertexAI = _Unavailable
    except Exception:
        llms_mod = types.ModuleType("langchain_community.llms")
        llms_mod.VertexAI = _Unavailable
        sys.modules["langchain_community.llms"] = llms_mod


def _markdown_table(ids: list[str], rows: list[dict[str, float]]) -> str:
    header = "| query | " + " | ".join(METRIC_NAMES) + " |"
    separator = "|" + "---|" * (len(METRIC_NAMES) + 1)
    body = []
    for query_id, row in zip(ids, rows, strict=True):
        cells = " | ".join(f"{row[metric]:.3f}" for metric in METRIC_NAMES)
        body.append(f"| {query_id} | {cells} |")
    average = {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRIC_NAMES}
    average_cells = " | ".join(f"{average[metric]:.3f}" for metric in METRIC_NAMES)
    body.append(f"| **average** | {average_cells} |")
    return "\n".join([header, separator, *body])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", help="JSON written by eval_ragas_s11.py --collect-only")
    parser.add_argument("--out", default=None, help="Write the scored report as JSON.")
    args = parser.parse_args()

    payload = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    samples = payload["samples"] if isinstance(payload, dict) else payload
    judge_model = (
        payload.get("judge_model", "gpt-4o-mini") if isinstance(payload, dict) else "gpt-4o-mini"
    )
    embedding_model = (
        payload.get("embedding_model", "text-embedding-3-small")
        if isinstance(payload, dict)
        else "text-embedding-3-small"
    )

    _install_vertex_shims()

    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_list(
        [
            {
                "question": sample["question"],
                "answer": sample["answer"],
                "contexts": sample["contexts"],
                "ground_truth": sample["ground_truth"],
            }
            for sample in samples
        ]
    )
    judge = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))

    print(f"Scoring {len(samples)} samples (judge={judge_model}, embeddings={embedding_model})...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embeddings,
    )

    frame = result.to_pandas()
    rows = [
        {
            metric: (float(frame.iloc[index][metric]) if metric in frame.columns else float("nan"))
            for metric in METRIC_NAMES
        }
        for index in range(len(samples))
    ]
    table = _markdown_table([sample["id"] for sample in samples], rows)

    print("\n=== RAGAS generation baseline ===")
    print(table)

    if args.out:
        report = {
            "judge_model": judge_model,
            "embedding_model": embedding_model,
            "table_markdown": table,
            "per_query": [
                {
                    "id": sample["id"],
                    **rows[index],
                    "citation_report": sample.get("citation_report"),
                }
                for index, sample in enumerate(samples)
            ],
        }
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
