"""Deriva el tier de audiencia en tiempo de ejecución a partir del contexto.

Los tiers moldean el encuadre del prompt (bullets ejecutivos vs hitos de PM vs
detalle de desarrollador vs por defecto). Dos formas de fijarlos:

- **Override explícito** del llamante (param API ``tier=executive``). Siempre gana.
- **Derivación implícita** a partir de la última transcripción + la metadata
  acumulada.

La derivación es una cadena de reglas puras ordenada por precedencia. Cada regla
devuelve ``True``/``False``; la primera coincidencia decide el tier y, si ninguna
coincide, el tier es ``DEFAULT``. El resolver devuelve también el **nombre de la
regla** que disparó — esa explicabilidad es lo que permite al panel lateral
mostrar "executive (nda_detected)" en lugar de solo "executive".
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import structlog

from app.sessions.models import ProjectMetadata

log = structlog.get_logger()


class Tier(StrEnum):
    EXECUTIVE = "executive"
    PM = "pm"
    DEVELOPER = "developer"
    DEFAULT = "default"


@dataclass
class ResolutionContext:
    transcript: str
    metadata: ProjectMetadata
    override: Tier | None = None


@dataclass
class TierRule:
    name: str
    tier: Tier
    predicate: Callable[[ResolutionContext], bool]


# --- Predicados --------------------------------------------------------------

_NDA_PATTERN = re.compile(
    r"\b(nda|non[- ]?disclosure|confidential|under embargo|legal hold)\b", re.IGNORECASE
)
_REGULATORY_PATTERN = re.compile(
    r"\b(hipaa|gdpr|sox|pci[- ]?dss|fda|iso[- ]?27001|ccpa)\b", re.IGNORECASE
)
_DEV_KEYWORDS_PATTERN = re.compile(
    r"\b(docker|kubernetes|k8s|microservice|microservices|terraform|iac|"
    r"helm|grpc|graphql|kafka|airflow|spark|rabbitmq)\b",
    re.IGNORECASE,
)


def _has_nda(ctx: ResolutionContext) -> bool:
    return bool(
        _NDA_PATTERN.search(ctx.transcript)
        or (ctx.metadata.agreed_scope and _NDA_PATTERN.search(ctx.metadata.agreed_scope))
    )


def _has_regulatory_context(ctx: ResolutionContext) -> bool:
    return bool(
        _REGULATORY_PATTERN.search(ctx.transcript)
        or (ctx.metadata.agreed_scope and _REGULATORY_PATTERN.search(ctx.metadata.agreed_scope))
        or any(_REGULATORY_PATTERN.search(tech) for tech in ctx.metadata.mentioned_technologies)
    )


def _is_small_team(ctx: ResolutionContext) -> bool:
    return ctx.metadata.assumed_team_size is not None and ctx.metadata.assumed_team_size <= 2


def _technical_audience(ctx: ResolutionContext) -> bool:
    hits = _DEV_KEYWORDS_PATTERN.findall(ctx.transcript)
    # Requiere al menos dos keywords técnicas distintas para promover — una
    # mención suelta ("we run on docker") es demasiado ruidosa.
    return len({t.lower() for t in hits}) >= 2


# --- Cadena de reglas --------------------------------------------------------

_RULES: tuple[TierRule, ...] = (
    TierRule("nda_detected", Tier.EXECUTIVE, _has_nda),
    TierRule("regulatory_context", Tier.EXECUTIVE, _has_regulatory_context),
    TierRule("technical_audience", Tier.DEVELOPER, _technical_audience),
    TierRule("low_budget_pm", Tier.PM, _is_small_team),
)


def resolve_tier(
    *,
    transcript: str,
    metadata: ProjectMetadata,
    override: Tier | None = None,
) -> tuple[Tier, str]:
    """Devuelve ``(tier, nombre_regla)``.

    Precedencia:
        1. ``override`` (del llamante) gana.
        2. Si no, se evalúa la cadena de reglas en orden — la primera gana.
        3. Si no, ``DEFAULT``.
    """
    ctx = ResolutionContext(transcript=transcript, metadata=metadata, override=override)

    if override is not None:
        log.info("tier_resolved", tier=override.value, rule="explicit_override")
        return override, "explicit_override"

    for rule in _RULES:
        try:
            if rule.predicate(ctx):
                log.info("tier_resolved", tier=rule.tier.value, rule=rule.name)
                return rule.tier, rule.name
        except Exception as exc:
            log.warning(
                "tier_rule_predicate_failed",
                rule=rule.name,
                error_type=type(exc).__name__,
                error=str(exc)[:120],
            )

    log.info("tier_resolved", tier=Tier.DEFAULT.value, rule="default")
    return Tier.DEFAULT, "default"
