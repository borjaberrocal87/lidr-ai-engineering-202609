"""Contratos Pydantic de la Sesión 9 (RAG end-to-end)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.generation.rag.schemas import (
    Assumption,
    BudgetComponent,
    Citation,
    EstimationModule,
    EstimationTask,
    GroundedEstimate,
    IngestRequest,
    ProjectBrief,
    RetrievalQuery,
    SourceReference,
    TaskHoursEstimate,
    TranscriptEstimateRequest,
)


def test_project_brief_defaults_are_empty_but_functionality_is_required() -> None:
    brief = ProjectBrief(functionality="B2B payments platform")

    assert brief.technologies == []
    assert brief.regulations == []
    assert brief.constraints == []
    assert brief.scale == "unknown"
    assert brief.sector is None


def test_transcript_request_rejects_short_transcripts_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TranscriptEstimateRequest(transcript="demasiado corto")

    with pytest.raises(ValidationError):
        TranscriptEstimateRequest(transcript="x" * 100, unexpected=True)


def test_transcript_request_accepts_idempotency_key() -> None:
    request = TranscriptEstimateRequest(transcript="x" * 100, idempotency_key="abc")

    assert request.idempotency_key == "abc"


def test_retrieval_query_bounds_top_k() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(query_text="histórico de pagos", top_k=31)


def test_grounded_estimate_parses_modules_tasks_and_citations() -> None:
    estimate = GroundedEstimate(
        total_engineer_days=12,
        modules=[
            EstimationModule(
                name="Payments",
                tasks=[
                    EstimationTask(name="Stripe integration", engineer_days=6, sources=[1, 2]),
                    EstimationTask(name="Reconciliation", engineer_days=6, sources=[2]),
                ],
            )
        ],
        duration_weeks=3,
        sources=[Citation(source_id=1, relevance="primary", used_for="gateway")],
        assumptions=[Assumption(description="PCI scope", impact="high", rationale="novel")],
        confidence="medium",
        reasoning="Dos fuentes históricas de pagos.",
    )

    assert estimate.modules[0].tasks[0].sources == [1, 2]
    assert estimate.total_engineer_days == 12


def test_budget_component_accepts_optional_module() -> None:
    component = BudgetComponent(
        component_id="AUTH-001",
        name="OAuth",
        description="Login",
        tech_stack=["python"],
        estimated_hours=10,
        complexity="medium",
        module="Authentication & Access",
    )

    assert component.module == "Authentication & Access"


def test_ingest_request_defaults_chunk_type_to_budget_component() -> None:
    request = IngestRequest(
        source_path="data/x.json::BUD-1",
        document_type="historical_budget",
        content={
            "budget_id": "BUD-1",
            "client_metadata": {"name": "Acme", "sector": "finance", "country": "ES"},
            "project_summary": "API",
            "main_technology": "python",
            "year": 2024,
            "total_estimated_hours": 10,
            "components": [
                {
                    "component_id": "AUTH-001",
                    "name": "OAuth",
                    "description": "Login",
                    "tech_stack": ["python"],
                    "estimated_hours": 10,
                    "complexity": "medium",
                }
            ],
        },
    )

    assert request.chunk_type == "budget_component"


def _source(chunk_id: str = "1", document_id: str = "BUD-1") -> SourceReference:
    return SourceReference(chunk_id=chunk_id, document_id=document_id, evidence="Component: OAuth")


def test_grounded_line_requires_a_citation() -> None:
    with pytest.raises(ValidationError):
        EstimationTask(name="OAuth", grounded=True)


def test_ungrounded_line_cannot_carry_hours_or_sources() -> None:
    with pytest.raises(ValidationError):
        EstimationTask(name="OAuth", grounded=False, engineer_days=5)

    with pytest.raises(ValidationError):
        EstimationTask(name="OAuth", grounded=False, citations=[_source()])


def test_grounded_line_keeps_legacy_sources_in_sync() -> None:
    task = EstimationTask(
        name="OAuth",
        engineer_days=5,
        grounded=True,
        citations=[_source("7"), _source("8")],
    )

    assert task.sources == [7, 8]


def test_legacy_task_with_only_sources_remains_valid() -> None:
    task = EstimationTask(name="OAuth", engineer_days=5, sources=[1, 2])

    assert task.grounded is True
    assert task.citations == []
    assert task.sources == [1, 2]


def test_task_hours_source_contract_is_consistent() -> None:
    legacy = TaskHoursEstimate(module="Auth", task="Login", has_match=True, estimated_hours=40)
    assert legacy.grounded is True

    no_match = TaskHoursEstimate(
        module="Auth", task="Login", has_match=False, grounded=False, sources=[]
    )
    assert no_match.estimated_hours is None

    with pytest.raises(ValidationError):
        TaskHoursEstimate(
            module="Auth",
            task="Login",
            has_match=True,
            grounded=True,
            sources=[],
        )
