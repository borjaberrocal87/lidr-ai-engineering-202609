"""Tests del orquestador Boss y de la pipeline Actor-Critic-Boss."""

from typing import Any

import pytest

from app.config import settings
from app.schemas.critic import CriticFeedback, CriticIssue
from app.schemas.estimations import DetailLevel, EstimationResult, OutputFormat, Phase, ProjectType
from app.services import estimation
from app.services.boss import Boss
from app.services.estimation import EstimationService
from app.sessions.models import ProjectMetadata, Session


def _result(*, confidence_pct: int = 72, cost: int = 25_000) -> EstimationResult:
    return EstimationResult(
        summary="Un CRM B2B de tamaño medio para el equipo de ventas.",
        confidence_pct=confidence_pct,
        phases=[
            Phase(
                name="Construcción",
                duration_weeks=6,
                cost_eur=cost,
                summary="Funcionalidad núcleo del CRM con React y Postgres.",
            )
        ],
        total_duration_weeks=6,
        total_cost_eur=cost,
    )


def _accept() -> CriticFeedback:
    return CriticFeedback(verdict="accept", issues=[], confidence_in_review=85)


def _needs_iteration() -> CriticFeedback:
    return CriticFeedback(
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


def _reject() -> CriticFeedback:
    return CriticFeedback(
        verdict="reject",
        issues=[
            CriticIssue(
                category="scope_mismatch",
                severity="critical",
                field_path="summary",
                description="El proyecto estimado no corresponde con lo pedido.",
            )
        ],
        confidence_in_review=70,
    )


# --- Boss (unit) ------------------------------------------------------------


def test_boss_accepts_on_first_iteration() -> None:
    boss = Boss(max_iterations=2)
    _draft, trace = boss.run(actor=lambda _f: _result(), critic=lambda _r: _accept())

    assert trace.final_decision == "accept"
    assert trace.iterations_run == 1
    assert trace.iterations[0].decision_after == "accept"


def test_boss_iterates_then_accepts_and_passes_feedback() -> None:
    reviews = iter([_needs_iteration(), _accept()])
    received_feedback: list[CriticFeedback | None] = []

    def actor(feedback: CriticFeedback | None) -> EstimationResult:
        received_feedback.append(feedback)
        return _result()

    boss = Boss(max_iterations=2)
    _draft, trace = boss.run(actor=actor, critic=lambda _r: next(reviews))

    assert trace.final_decision == "accept"
    assert trace.iterations_run == 2
    assert received_feedback[0] is None
    assert received_feedback[1] is not None
    assert received_feedback[1].verdict == "needs_iteration"


def test_boss_synthesizes_on_reject() -> None:
    boss = Boss(max_iterations=2)
    result, trace = boss.run(actor=lambda _f: _result(), critic=lambda _r: _reject())

    assert trace.final_decision == "synthesize"
    assert trace.iterations[0].decision_after == "synthesize"
    assert "Open caveats" in result.summary
    # 72 // 2 = 36, por encima del suelo de 30.
    assert result.confidence_pct == 36


def test_boss_synthesizes_when_iteration_budget_exhausted() -> None:
    boss = Boss(max_iterations=1)
    result, trace = boss.run(
        actor=lambda _f: _result(confidence_pct=80),
        critic=lambda _r: _needs_iteration(),
    )

    assert trace.final_decision == "synthesize"
    assert trace.iterations_run == 1
    assert result.confidence_pct == 40


def test_boss_rejects_non_positive_iterations() -> None:
    with pytest.raises(ValueError):
        Boss(max_iterations=0)


# --- EstimationService.estimate_with_acb (integración) ----------------------


class _ACBFakeWrapper:
    """Doble que distingue estimación (actor) de feedback (critic) por schema."""

    def __init__(self, drafts: list[EstimationResult], feedbacks: list[CriticFeedback]):
        self.drafts = drafts
        self.feedbacks = feedbacks
        self.estimation_calls: list[list[dict[str, str]]] = []
        self.critic_calls: list[list[dict[str, str]]] = []

    def complete_structured_messages(
        self, *, messages: list[dict[str, str]], response_model: type, **kwargs: Any
    ) -> tuple[Any, dict[str, Any]]:
        if response_model is CriticFeedback:
            self.critic_calls.append(messages)
            index = min(len(self.critic_calls) - 1, len(self.feedbacks) - 1)
            return self.feedbacks[index], _meta()
        self.estimation_calls.append(messages)
        index = min(len(self.estimation_calls) - 1, len(self.drafts) - 1)
        return self.drafts[index], _meta()

    def complete_structured(self, **kwargs: Any) -> tuple[ProjectMetadata, dict[str, Any]]:
        return ProjectMetadata(), _meta()


def _meta() -> dict[str, Any]:
    return {"model": "gpt-4o-mini", "provider": "openai"}


def test_estimate_with_acb_returns_audit_trail(monkeypatch) -> None:
    fake = _ACBFakeWrapper(drafts=[_result()], feedbacks=[_accept()])
    monkeypatch.setattr(estimation, "get_llm_wrapper", lambda: fake)
    monkeypatch.setattr(settings, "guardrails_enabled", False)

    response = EstimationService().estimate_with_acb(
        session=Session(),
        transcript="We want a CRM called Nimbus built with React for the sales team.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )

    assert response.acb.final_decision == "accept"
    assert response.acb.iterations_run == 1
    assert response.result.total_cost_eur == 25_000


def test_estimate_with_acb_persists_single_turn(monkeypatch) -> None:
    fake = _ACBFakeWrapper(
        drafts=[_result(), _result(confidence_pct=88)],
        feedbacks=[_needs_iteration(), _accept()],
    )
    monkeypatch.setattr(estimation, "get_llm_wrapper", lambda: fake)
    monkeypatch.setattr(settings, "guardrails_enabled", False)

    session = Session()
    response = EstimationService().estimate_with_acb(
        session=session,
        transcript="We want a CRM called Nimbus built with React for the sales team.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )

    # Dos rondas actor+critic, pero un único turno persistido en la sesión.
    assert response.acb.iterations_run == 2
    assert len(session.history.messages) == 2
    assert len(fake.estimation_calls) == 2


def test_estimate_with_acb_explicit_tier_override(monkeypatch) -> None:
    from app.sessions.tier_resolver import Tier

    fake = _ACBFakeWrapper(drafts=[_result()], feedbacks=[_accept()])
    monkeypatch.setattr(estimation, "get_llm_wrapper", lambda: fake)
    monkeypatch.setattr(settings, "guardrails_enabled", False)

    session = Session()
    EstimationService().estimate_with_acb(
        session=session,
        transcript="Queremos un CRM para ventas.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        tier=Tier.EXECUTIVE,
    )

    assert session.last_resolved_tier == "executive"
    assert session.last_tier_rule == "explicit_override"
