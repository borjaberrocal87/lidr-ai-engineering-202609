"""Typed contracts of the estimation graph (Session 13).

Three families:

* **Structured LLM output** — ``RequirementsExtraction`` and
  ``ComponentClassification`` are what ``LLMWrapper.complete_structured``
  validates with Instructor.
* **Estimate** — ``EstimateReport`` / ``EstimateComponent`` are built
  deterministically in engineer-hours (the Session 12 cost model), so the graph
  only makes two LLM calls per run.
* **HTTP contract** — ``GraphEstimateRequest`` and ``GraphEstimateResponse`` keep
  the same external shape as the other estimators (transcript in, structured
  estimate + ``status`` out) plus the rendered ``run_text``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

GraphStatus = Literal["validated", "needs_review"]


# --------------------------------------------------------------------------- #
# Structured LLM output                                                        #
# --------------------------------------------------------------------------- #
class ComponentOut(BaseModel):
    """One functional component with its coarse category."""

    name: str = Field(min_length=1, description="Short component name.")
    category: str = Field(
        min_length=1,
        description="Coarse category, e.g. backend, integration, mobile, frontend, analytics.",
    )


class RequirementsExtraction(BaseModel):
    """Flat list of concrete requirements extracted from a transcript."""

    requirements: list[str] = Field(
        default_factory=list,
        description="One atomic technical requirement per item, in English.",
    )


class ComponentClassification(BaseModel):
    """Grouping of requirements into the components needed to deliver them."""

    components: list[ComponentOut] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Estimate (deterministic, engineer-hours)                                     #
# --------------------------------------------------------------------------- #
class EstimateComponent(BaseModel):
    """One costed component in the estimate."""

    name: str = Field(min_length=1)
    estimated_hours: float = Field(ge=0)
    unbudgeted: bool = Field(
        description="True when no historical reference was found; hours are zero.",
    )
    reference_budget_ids: list[str] = Field(default_factory=list)
    rationale: str


class EstimateReport(BaseModel):
    """Deterministic estimate in engineer-hours."""

    components: list[EstimateComponent] = Field(default_factory=list)
    total_hours: float = Field(ge=0)


# --------------------------------------------------------------------------- #
# Observability                                                                #
# --------------------------------------------------------------------------- #
class NodeRunRecord(BaseModel):
    """One node execution, visible to the UI and the CLI timeline."""

    name: str
    duration_ms: float = Field(ge=0)


# --------------------------------------------------------------------------- #
# HTTP contract                                                                #
# --------------------------------------------------------------------------- #
class GraphEstimateRequest(BaseModel):
    """Payload of ``POST /api/v1/graph/estimate/from-transcript``."""

    model_config = ConfigDict(extra="forbid")

    transcript: str = Field(min_length=100, max_length=50_000)
    estimation_id: str | None = Field(
        default=None,
        description="Thread id for the checkpointer; a fresh uuid4 is used when omitted.",
    )


class BudgetMatchView(BaseModel):
    """A retrieved historical reference, as returned to the caller."""

    component: str
    reference_budget_id: str | None = None
    amount: float
    distance: float


class GraphEstimateResponse(BaseModel):
    """Estimate + status + the state needed to render the run."""

    estimation_id: str
    status: GraphStatus | None
    estimate: dict[str, Any] | None = None
    requirements: list[str] = Field(default_factory=list)
    components: list[ComponentOut] = Field(default_factory=list)
    budget_matches: list[BudgetMatchView] = Field(default_factory=list)
    node_runs: list[NodeRunRecord] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    run_text: str = Field(description="Rendered run summary, ready to download.")
