#!/usr/bin/env python3
"""Session 12 - run the hand-written estimation agent over a transcript.

This is the deliverable generator: it feeds a meeting transcript to the manual
agent loop (``app/generation/agentic/estimation_agent/runner.py``), prints the
step-by-step trace (reasoning -> action -> observation) and the final structured
report, and optionally writes them to a file.

Cost discipline: debug the LOOP MECHANICS cheaply first with ``gpt-5-mini`` and
the simple transcript, then run ``gpt-5`` / ``medium`` on the complex transcript
for the deliverable.

    # 1) offline loop debugging with the stub backend (NO database needed)
    uv run python scripts/run_agent_s12.py \\
        examples/session-12/sample_transcript_simple.txt \\
        --model gpt-5-mini --effort minimal --stub

    # 2) real retrieval, cheap model
    docker compose exec api python scripts/run_agent_s12.py \\
        examples/session-12/sample_transcript_simple.txt \\
        --model gpt-5-mini --effort minimal

    # 3) the deliverable run
    docker compose exec api python scripts/run_agent_s12.py \\
        examples/session-12/sample_transcript_complex.txt \\
        --model gpt-5 --effort medium \\
        --out examples/session-12/trace_complex.txt

``search_budgets`` wraps the real S9-S10 ``retrieve()`` pipeline by default, so
real runs need the stack up and both corpora ingested. ``--stub`` swaps in the
offline fixture from ``examples/session-12/offline_budget_stub.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.dependencies import (  # noqa: E402
    get_async_openai_client,
    get_budget_search_backend,
)
from app.generation.agentic.estimation_agent.runner import run_estimation_agent  # noqa: E402
from app.generation.agentic.estimation_agent.schemas import (  # noqa: E402
    AgentRunResult,
    SearchBudgetsArgs,
)
from app.generation.agentic.estimation_agent.tools import BudgetSearchBackend  # noqa: E402

STUB_PATH = ROOT / "examples" / "session-12" / "offline_budget_stub.py"


def _load_stub_backend() -> BudgetSearchBackend:
    """Load the offline fixture and adapt it to the async backend protocol."""
    spec = importlib.util.spec_from_file_location("s12_offline_budget_stub", STUB_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load the offline stub from {STUB_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def backend(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
        filters = args.filters.model_dump() if args.filters else None
        return list(module.search_budgets_stub(args.query, filters))

    return backend


def _render(result: AgentRunResult) -> str:
    lines = [result.trace.render(), ""]
    header = f"FINAL REPORT  (iterations={result.iterations}, stopped={result.stopped_reason})"
    lines.extend(["=" * 78, header, "=" * 78])
    report = result.report
    if report is None:
        lines.append("(the agent stopped without producing a structured report)")
        return "\n".join(lines)

    for component in report.components:
        sources = ", ".join(str(source) for source in component.source_chunk_ids) or "none"
        lines.append(f"  - {component.name}: {component.estimated_hours}h  [sources: {sources}]")
        lines.append(f"      {component.rationale}")
    lines.extend(["", f"  TOTAL: {report.total_hours}h    confidence: {report.confidence}"])
    if report.assumptions:
        lines.append("  assumptions:")
        lines.extend(f"    - {assumption}" for assumption in report.assumptions)
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> int:
    transcript_path = Path(args.transcript)
    if not transcript_path.is_file():
        print(f"ERROR: transcript not found: {transcript_path}", file=sys.stderr)
        return 1

    client = get_async_openai_client()
    if client is None:
        print(
            "ERROR: no OpenAI key configured - the agent needs the Responses API "
            "(set OPEN_AI_KEY in .env).",
            file=sys.stderr,
        )
        return 1

    if args.stub:
        backend: BudgetSearchBackend | None = _load_stub_backend()
    else:
        backend = get_budget_search_backend()
    if backend is None:
        print(
            "ERROR: the retrieval backend is not available (no embeddings configured). "
            "Ingest the corpus or use --stub.",
            file=sys.stderr,
        )
        return 1

    transcript = transcript_path.read_text(encoding="utf-8")
    print(f"transcript : {transcript_path}")
    print(
        f"model      : {args.model}   effort: {args.effort}   backend: "
        f"{'offline stub' if args.stub else 'retrieve() pipeline'}"
    )
    print()

    result = await run_estimation_agent(
        transcript,
        client=client,
        model=args.model,
        reasoning_effort=args.effort,
        max_iterations=args.max_iterations,
        backend=backend,
    )

    rendered = _render(result)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
        print(f"\n(trace written to {args.out})")
    return 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the Session 12 estimation agent.")
    parser.add_argument("transcript", help="Path to a meeting transcript .txt file.")
    parser.add_argument(
        "--model",
        default=settings.agent_model,
        help=f"OpenAI model (default {settings.agent_model}).",
    )
    parser.add_argument(
        "--effort",
        default=settings.agent_reasoning_effort,
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort for the Responses API.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=settings.agent_max_iterations,
        help="Loop safeguard: max Responses API round-trips.",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use the offline budget fixture (no database or corpus needed).",
    )
    parser.add_argument("--out", help="Write the rendered trace and report to this file.")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
