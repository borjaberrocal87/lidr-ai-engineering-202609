"""Tests del resolver de tier dinámico (sesión 5, directo)."""

from app.sessions.models import ProjectMetadata
from app.sessions.tier_resolver import Tier, resolve_tier


def _resolve(
    transcript: str, metadata: ProjectMetadata | None = None, override: Tier | None = None
):
    return resolve_tier(
        transcript=transcript, metadata=metadata or ProjectMetadata(), override=override
    )


def test_default_tier_when_no_rule_matches() -> None:
    tier, rule = _resolve("Queremos un CRM sencillo para el equipo comercial.")

    assert tier is Tier.DEFAULT
    assert rule == "default"


def test_nda_detected_maps_to_executive() -> None:
    tier, rule = _resolve("Antes de empezar, firmamos un NDA con el cliente.")

    assert tier is Tier.EXECUTIVE
    assert rule == "nda_detected"


def test_regulatory_context_maps_to_executive() -> None:
    tier, rule = _resolve("El sistema debe cumplir GDPR y retener datos siete años.")

    assert tier is Tier.EXECUTIVE
    assert rule == "regulatory_context"


def test_technical_audience_maps_to_developer() -> None:
    transcript = "Desplegamos en Kubernetes con Kafka y gRPC entre microservicios."

    tier, rule = _resolve(transcript)

    assert tier is Tier.DEVELOPER
    assert rule == "technical_audience"


def test_single_technical_keyword_does_not_promote() -> None:
    tier, _ = _resolve("Lo correremos en docker.")

    assert tier is Tier.DEFAULT


def test_small_team_maps_to_pm() -> None:
    tier, rule = _resolve(
        "Proyecto pequeño.",
        ProjectMetadata(assumed_team_size=2),
    )

    assert tier is Tier.PM
    assert rule == "low_budget_pm"


def test_explicit_override_wins_over_rules() -> None:
    tier, rule = _resolve(
        "Firmamos un NDA y usamos Kubernetes con Kafka.",
        override=Tier.PM,
    )

    assert tier is Tier.PM
    assert rule == "explicit_override"


def test_nda_precedence_over_technical() -> None:
    transcript = "Hay un NDA firmado y desplegamos en Kubernetes con Kafka."

    tier, rule = _resolve(transcript)

    assert tier is Tier.EXECUTIVE
    assert rule == "nda_detected"
