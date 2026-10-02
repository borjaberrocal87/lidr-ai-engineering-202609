"""Tests del guardrail de salida `enforce_scope_response` (política filter)."""

from app.guardrails.output import enforce_scope_response
from app.schemas.estimations import EstimationResult, Phase


def _high_confidence() -> EstimationResult:
    return EstimationResult(
        summary="Una landing page con CRM, blog y formulario de contacto.",
        confidence_pct=70,
        phases=[
            Phase(
                name="Discovery",
                duration_weeks=1,
                cost_eur=2000,
                summary="Alcance y entrevistas con marketing.",
            )
        ],
        total_duration_weeks=1,
        total_cost_eur=2000,
    )


def test_high_confidence_is_unchanged() -> None:
    result = _high_confidence()

    assert enforce_scope_response(result) is result


def test_already_marked_is_unchanged() -> None:
    result = EstimationResult(
        summary="Out of scope: falta información de alcance.",
        confidence_pct=10,
        phases=[
            Phase(
                name="Not estimated",
                duration_weeks=1,
                cost_eur=0,
                summary="Sin información suficiente para dimensionar.",
            )
        ],
        total_duration_weeks=1,
        total_cost_eur=0,
    )

    assert enforce_scope_response(result) is result


def test_low_confidence_without_prefix_is_filtered() -> None:
    # `model_construct` evita los validadores para simular el borde del filtro.
    raw = EstimationResult.model_construct(
        summary="No estoy seguro de las cifras.",
        confidence_pct=10,
        phases=[Phase(name="X", duration_weeks=1, cost_eur=0, summary="Dudoso, falta contexto.")],
        total_duration_weeks=1,
        total_cost_eur=0,
    )

    filtered = enforce_scope_response(raw)

    assert filtered.summary.startswith("Out of scope:")
    assert filtered.total_cost_eur == 0
    assert filtered.phases[0].name == "Not estimated"
