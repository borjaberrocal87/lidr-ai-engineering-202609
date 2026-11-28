#!/usr/bin/env python3
"""Session 13 - run the estimation graph over a transcript, end to end.

Drives the compiled LangGraph ``StateGraph``
(``app/generation/agentic/estimation_graph``) sequentially:

    extract_requirements -> classify_components -> search_budgets
      -> generate_estimate -> validate_and_consolidate

and prints the final state (node timeline -> requirements -> components ->
budget matches -> estimate -> status). This is the script that produces the
deliverable run/trace.

Persistence + observability (Levels 2):

* By default it opens the SAME Postgres the project uses (pgvector) as the
  checkpointer; pass ``--memory`` to use an in-process ``MemorySaver`` instead.
* Set ``LOGFIRE_TOKEN`` to export one span per node to Pydantic Logfire and get
  the trace link. With no token the spans run locally and export nothing; the
  node timeline still comes from the ``node_runs`` accumulator.

Cost discipline: debug the mechanics first with ``--memory --stub`` and the
simple transcript (only two cheap LLM calls, no Docker/Postgres); the deliverable
run uses the real checkpointer + retrieval on the complex transcript:

    # offline mechanics smoke (needs OPENAI_API_KEY only)
    uv run python scripts/run_graph_s13.py --memory --stub \\
        --transcript examples/session-12/sample_transcript_simple.txt

    # deliverable run (stack up + both corpora ingested)
    docker compose exec api python scripts/run_graph_s13.py \\
        --transcript examples/session-12/sample_transcript_complex.txt \\
        --out examples/session-13/trace_complex.txt
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
from app.dependencies import get_llm_wrapper, make_budget_search_backend  # noqa: E402
from app.generation.agentic.estimation_agent.schemas import SearchBudgetsArgs  # noqa: E402
from app.generation.agentic.estimation_agent.tools import BudgetSearchBackend  # noqa: E402
from app.generation.agentic.estimation_graph.build import build_graph  # noqa: E402
from app.generation.agentic.estimation_graph.observability import configure_logfire  # noqa: E402
from app.generation.agentic.estimation_graph.render import render_run  # noqa: E402

DEFAULT_TRANSCRIPT = ROOT / "examples" / "session-12" / "sample_transcript_complex.txt"
STUB_PATH = ROOT / "examples" / "session-12" / "offline_budget_stub.py"


def _load_stub_backend() -> BudgetSearchBackend:
    """Wrap the Session 12 offline retrieval fixture as a BudgetSearchBackend."""
    spec = importlib.util.spec_from_file_location("s13_offline_budget_stub", STUB_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load the offline stub from {STUB_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def backend(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
        filters = None
        if args.filters and (args.filters.sectors or args.filters.component_type):
            filters = {
                "sectors": args.filters.sectors,
                "component_type": args.filters.component_type,
            }
        return module.search_budgets_stub(args.query, filters)

    return backend


async def _run(graph: Any, transcript: str, estimation_id: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": estimation_id}}
    return await graph.ainvoke(
        {"transcript": transcript, "estimation_id": estimation_id},
        config,
    )


async def _main_async(args: argparse.Namespace) -> int:
    transcript_path = Path(args.transcript)
    if not transcript_path.is_file():
        print(f"ERROR: transcript not found: {transcript_path}", file=sys.stderr)
        return 1
    transcript = transcript_path.read_text(encoding="utf-8")
    estimation_id = args.estimation_id or f"s13-{transcript_path.stem}"

    configure_logfire()  # no FastAPI app in the CLI; spans + httpx only.

    settings = get_settings()
    if args.stub:
        backend: BudgetSearchBackend | None = _load_stub_backend()
    else:
        backend = make_budget_search_backend(
            top_k=settings.graph_search_top_k,
            distance_threshold=settings.graph_search_distance_threshold,
        )
    if backend is None:
        print(
            "ERROR: no budget search backend (missing embeddings configuration).",
            file=sys.stderr,
        )
        return 1

    wrapper = get_llm_wrapper()
    print(f"transcript    : {transcript_path}")
    print(f"checkpointer  : {'MemorySaver' if args.memory else 'AsyncPostgresSaver'}")
    print(f"retrieval     : {'stub (offline)' if args.stub else 'real retrieve()'}")
    print(f"estimation_id : {estimation_id}\n")

    if args.memory:
        from langgraph.checkpoint.memory import MemorySaver

        graph = build_graph(
            llm_wrapper=wrapper,
            backend=backend,
            checkpointer=MemorySaver(),
        )
        state = await _run(graph, transcript, estimation_id)
    else:
        from app.generation.agentic.estimation_graph.checkpointer import open_checkpointer

        async with open_checkpointer() as checkpointer:
            graph = build_graph(
                llm_wrapper=wrapper,
                backend=backend,
                checkpointer=checkpointer,
            )
            state = await _run(graph, transcript, estimation_id)

    rendered = render_run(state)
    print(rendered)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"\n(run written to {out_path})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Session 13 estimation graph.")
    parser.add_argument(
        "--transcript",
        default=str(DEFAULT_TRANSCRIPT),
        help="Path to a meeting transcript .txt (default: the complex one from S12).",
    )
    parser.add_argument(
        "--estimation-id",
        help="thread_id for the checkpointer (default derived from the transcript name).",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Use an in-process MemorySaver instead of the Postgres checkpointer.",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use the offline retrieval stub (no database for search_budgets).",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "examples" / "session-13" / "trace_complex.txt"),
        help="Write the rendered run to this file.",
    )
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
