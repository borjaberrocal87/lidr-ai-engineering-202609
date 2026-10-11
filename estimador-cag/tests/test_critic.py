"""Tests del servicio Critic (patrón Actor-Critic-Boss)."""

from typing import Any

from app.schemas.critic import CriticFeedback, CriticIssue
from app.schemas.estimations import EstimationResult, Phase
from app.services.critic import Critic
from app.sessions.models import ProjectMetadata
from app.sessions.tier_resolver import Tier


def _result() -> EstimationResult:
    return EstimationResult(
        summary="Un CRM B2B de tamaño medio para el equipo de ventas.",
        confidence_pct=72,
        phases=[
            Phase(
                name="Construcción",
                duration_weeks=6,
                cost_eur=25_000,
                summary="Funcionalidad núcleo del CRM con React y Postgres.",
            )
        ],
        total_duration_weeks=6,
        total_cost_eur=25_000,
    )


def _meta() -> dict[str, Any]:
    return {"model": "gpt-4o-mini", "provider": "openai", "latency_ms": 12.0}


class _FakeWrapper:
    """Doble del wrapper: devuelve un `CriticFeedback` fijo o lanza un error."""

    def __init__(self, feedback: CriticFeedback | None = None, error: Exception | None = None):
        self.feedback = feedback or CriticFeedback(
            verdict="accept", issues=[], confidence_in_review=80
        )
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def complete_structured_messages(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.feedback, _meta()


def _review(wrapper: _FakeWrapper) -> CriticFeedback:
    critic = Critic(llm_wrapper=wrapper, model="gpt-4o-mini")
    return critic.review(
        transcript="Queremos un CRM para ventas con React.",
        metadata=ProjectMetadata(project_name="Nimbus"),
        tier=Tier.PM,
        result=_result(),
    )


def test_critic_returns_feedback_on_success() -> None:
    feedback = CriticFeedback(
        verdict="needs_iteration",
        issues=[
            CriticIssue(
                category="phase_imbalance",
                severity="major",
                field_path="phases[0].cost_eur",
                description="Una sola fase concentra todo el coste sin justificar.",
            )
        ],
        confidence_in_review=60,
    )
    wrapper = _FakeWrapper(feedback=feedback)

    result = _review(wrapper)

    assert result == feedback
    assert wrapper.calls[0]["response_model"] is CriticFeedback
    assert wrapper.calls[0]["model_override"] == "gpt-4o-mini"


def test_critic_failure_degrades_to_accept_with_zero_confidence() -> None:
    wrapper = _FakeWrapper(error=RuntimeError("boom"))

    result = _review(wrapper)

    assert result.verdict == "accept"
    assert result.issues == []
    assert result.confidence_in_review == 0
