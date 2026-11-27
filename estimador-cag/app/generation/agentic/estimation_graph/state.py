"""Typed, shared state threaded through the estimation graph (Session 13).

A LangGraph ``StateGraph`` passes ONE state object through every node; each node
returns a *partial* update and LangGraph merges it in. Most fields merge
last-writer-wins; a field annotated with a **reducer** delegates the merge to
that reducer instead. Two accumulators use ``operator.add``:

* ``budget_matches`` — ``search_budgets`` walks the components one at a time and
  returns only that component's matches, so the list grows along the flow (and,
  once the live session parallelises the search, across concurrent branches).
* ``errors`` — any node may append a soft failure without clobbering earlier ones.

``node_runs`` is a third accumulator owned by the tracing helper: one record per
node with its duration, so the UI and the CLI can render a timeline without
depending on an external observability backend.

``total=False`` so a node may return a partial dict without every key present and
the initial ``ainvoke`` only supplies ``transcript`` (+ ``estimation_id``).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class Component(TypedDict):
    """One functional component the project decomposes into."""

    name: str
    category: str


class BudgetMatch(TypedDict):
    """A historical reference budget retrieved for a component.

    ``amount`` carries the matched item's recorded engineer-hours (the grounding
    number the estimate is built from); ``distance`` is its cosine distance from
    the query (lower = closer).
    """

    component: str
    reference_budget_id: str | None
    amount: float
    distance: float


class NodeRun(TypedDict):
    """Observability record for one node execution."""

    name: str
    duration_ms: float


class EstimationState(TypedDict, total=False):
    """The state threaded through the graph."""

    transcript: str
    estimation_id: str
    requirements: list[str]
    components: list[Component]
    # Accumulators: a node returns only the NEW items; the reducer appends.
    budget_matches: Annotated[list[BudgetMatch], operator.add]
    node_runs: Annotated[list[NodeRun], operator.add]
    estimate: dict[str, Any] | None
    status: str | None  # "validated" | "needs_review"
    errors: Annotated[list[str], operator.add]
