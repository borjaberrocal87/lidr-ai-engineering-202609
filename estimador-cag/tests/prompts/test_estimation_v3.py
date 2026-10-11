"""Tests del prompt conversacional v3 (bloque <audiencia> + critic_feedback)."""

from app.prompts.loader import available_estimation_versions, render_conversational_prompt
from app.schemas.critic import CriticFeedback, CriticIssue
from app.schemas.estimations import DetailLevel, OutputFormat, ProjectType
from app.sessions.models import ProjectMetadata
from app.sessions.tier_resolver import Tier


def _render(tier: object | None = None, critic_feedback: object | None = None) -> tuple[str, str]:
    return render_conversational_prompt(
        description="Queremos un CRM para el equipo de ventas con React.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        metadata=ProjectMetadata(project_name="Nimbus"),
        version="v3",
        tier=tier,
        critic_feedback=critic_feedback,
    )


def test_v3_is_available_on_disk() -> None:
    assert "v3" in available_estimation_versions()


def test_v3_director_tier_includes_audience_block() -> None:
    system, _ = _render(tier=Tier.EXECUTIVE)

    assert "<audiencia>" in system
    assert '"executive"' in system
    assert "dirección" in system.lower()


def test_v3_developer_tier_mentions_technical_detail() -> None:
    system, _ = _render(tier=Tier.DEVELOPER)

    assert "<audiencia>" in system
    assert '"developer"' in system
    assert "tech lead" in system.lower()


def test_v3_without_tier_omits_audience_block() -> None:
    system, _ = _render(tier=None)

    assert "<audiencia>" not in system


def test_v3_includes_critic_feedback_in_user_prompt() -> None:
    feedback = CriticFeedback(
        verdict="needs_iteration",
        issues=[
            CriticIssue(
                category="phase_imbalance",
                severity="major",
                field_path="phases[0].cost_eur",
                description="Una sola fase concentra todo el coste.",
                suggested_fix="Reparte el coste en más fases.",
            )
        ],
        confidence_in_review=60,
    )

    _, user = _render(tier=Tier.PM, critic_feedback=feedback)

    assert "<critic_feedback>" in user
    assert "phase_imbalance" in user
    assert "Reparte el coste" in user


def test_v3_without_critic_feedback_omits_block() -> None:
    _, user = _render(tier=Tier.PM, critic_feedback=None)

    assert "<critic_feedback>" not in user
