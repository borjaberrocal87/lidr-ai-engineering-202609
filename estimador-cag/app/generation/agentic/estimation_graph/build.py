"""Wire and compile the estimation graph (Session 13, Level 1 + Level 3).

Five nodes, sequential for now (the live session parallelises ``search_budgets``
with the Send API):

    START → extract_requirements → classify_components → search_budgets
          → generate_estimate → validate_and_consolidate → (conditional) → END

**Level 3** — the fixed ``validate_and_consolidate → END`` edge is replaced by a
CONDITIONAL edge: ``route_on_status`` reads the ``status`` the validation node set
and routes ``"validated"`` vs ``"needs_review"``. Both currently terminate at END
(the serious retry / fallback / HITL branches are built live); the routing already
makes the branch explicit.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.foundation.llm.wrapper import LLMWrapper
from app.generation.agentic.estimation_agent.tools import BudgetSearchBackend
from app.generation.agentic.estimation_graph.nodes import (
    make_classify_components_node,
    make_extract_requirements_node,
    make_generate_estimate_node,
    make_search_budgets_node,
    make_validate_and_consolidate_node,
)
from app.generation.agentic.estimation_graph.state import EstimationState


def route_on_status(state: EstimationState) -> str:
    """Conditional-edge router: branch on the validation ``status`` (Level 3)."""
    return "needs_review" if state.get("status") == "needs_review" else "validated"


def build_graph(
    *,
    llm_wrapper: LLMWrapper,
    backend: BudgetSearchBackend,
    checkpointer: Any | None = None,
) -> Any:
    """Build and compile the estimation graph.

    ``checkpointer`` persists state per ``thread_id`` (an ``AsyncPostgresSaver`` in
    the app, a ``MemorySaver`` in tests). ``None`` compiles a checkpointer-less
    graph — still runnable, just not resumable.
    """
    settings = get_settings()
    extract = make_extract_requirements_node(
        llm_wrapper,
        model=settings.graph_extraction_model,
        max_tokens=settings.graph_extraction_max_tokens,
        temperature=settings.temperature,
    )
    classify = make_classify_components_node(
        llm_wrapper,
        model=settings.graph_extraction_model,
        max_tokens=settings.graph_extraction_max_tokens,
        temperature=settings.temperature,
    )

    builder = StateGraph(EstimationState)
    # ``cast`` keeps mypy happy with LangGraph's generic node protocol; the node
    # signatures are enforced by ``NodeFn`` in ``nodes.py``.
    builder.add_node("extract_requirements", cast(Any, extract))
    builder.add_node("classify_components", cast(Any, classify))
    builder.add_node("search_budgets", cast(Any, make_search_budgets_node(backend)))
    builder.add_node("generate_estimate", cast(Any, make_generate_estimate_node()))
    builder.add_node(
        "validate_and_consolidate",
        cast(Any, make_validate_and_consolidate_node()),
    )

    builder.add_edge(START, "extract_requirements")
    builder.add_edge("extract_requirements", "classify_components")
    builder.add_edge("classify_components", "search_budgets")
    builder.add_edge("search_budgets", "generate_estimate")
    builder.add_edge("generate_estimate", "validate_and_consolidate")
    # Level 3: conditional exit keyed on the validation status.
    builder.add_conditional_edges(
        "validate_and_consolidate",
        route_on_status,
        {"validated": END, "needs_review": END},
    )

    return builder.compile(checkpointer=checkpointer)
