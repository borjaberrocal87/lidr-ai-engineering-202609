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
