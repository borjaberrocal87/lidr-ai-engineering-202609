"""Guardas de grounding: citas inventadas y coherencia de insuficiencia."""

from __future__ import annotations

from app.generation.rag.grounding import find_fabricated_citations, is_coherent_estimate
from app.generation.rag.schemas import (
    Citation,
    EstimationModule,
    EstimationTask,
    GroundedEstimate,
    RetrievedChunk,
)


def _chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id=1,
        content="histórico",
        sector="finance",
        project_year=2024,
        chunk_type="budget_component",
        distance=0.2,
    )


def _estimate(
    *,
    cites: list[int],
    task_cites: list[int],
    confidence: str = "medium",
) -> GroundedEstimate:
    return GroundedEstimate(
        total_engineer_days=10 if confidence != "insufficient" else None,
        modules=[
            EstimationModule(
                name="Auth",
                tasks=[EstimationTask(name="OAuth", engineer_days=10, sources=task_cites)],
            )
        ]
        if confidence != "insufficient"
        else [],
        duration_weeks=2 if confidence != "insufficient" else None,
        sources=[
            Citation(source_id=source_id, relevance="primary", used_for="auth")
            for source_id in cites
        ],
        assumptions=[],
        confidence=confidence,  # type: ignore[arg-type]
        reasoning="derivada de fuentes",
        insufficient_context_explanation=(
            "faltan presupuestos comparables" if confidence == "insufficient" else None
        ),
    )


def test_no_fabricated_ids_when_all_citations_exist() -> None:
    estimate = _estimate(cites=[1], task_cites=[2])

    assert find_fabricated_citations(estimate, [_chunk(1), _chunk(2)]) == []


def test_detects_fabricated_top_level_and_task_ids() -> None:
    estimate = _estimate(cites=[1, 99], task_cites=[2, 100])

    assert find_fabricated_citations(estimate, [_chunk(1), _chunk(2)]) == [99, 100]


def test_insufficient_estimate_is_coherent_only_without_numbers_or_modules() -> None:
    coherent = _estimate(cites=[], task_cites=[], confidence="insufficient")

    assert is_coherent_estimate(coherent) is True


def test_insufficient_estimate_with_numbers_is_incoherent() -> None:
    estimate = _estimate(cites=[1], task_cites=[1], confidence="insufficient").model_copy(
        update={"total_engineer_days": 4}
    )

    assert is_coherent_estimate(estimate) is False


def test_any_other_confidence_is_coherent() -> None:
    assert is_coherent_estimate(_estimate(cites=[1], task_cites=[1], confidence="high")) is True
