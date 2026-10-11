"""Esquema de feedback estructurado que produce el Critic.

El esquema es **el** contrato entre Critic y Boss. El Boss lee ``category``,
``severity`` y ``field_path`` de cada issue para decidir qué hacer (aceptar la
salida del actor, iterar con feedback o sintetizar un fallback). Revisiones en
texto libre obligarían al Boss a parsear prosa — una clase de bugs que evitamos
a propósito haciendo el Critic estructurado desde el primer día.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

CriticIssueCategory = Literal[
    "math_error",
    "hallucination",
    "scope_mismatch",
    "phase_imbalance",
    "missing_assumption",
    "unrealistic_estimate",
    "tier_mismatch",
]


CriticIssueSeverity = Literal["critical", "major", "minor"]


CriticVerdict = Literal["accept", "needs_iteration", "reject"]


class CriticIssue(BaseModel):
    """Un defecto concreto señalado por el Critic.

    ``field_path`` usa notación con puntos/corchetes referida a la estimación
    revisada (``phases[2].cost_eur``, ``summary``, ``total_cost_eur``). El Boss
    puede reenviarlo tal cual al prompt de la siguiente iteración del actor para
    que el modelo sepa qué cambiar.
    """

    category: CriticIssueCategory
    severity: CriticIssueSeverity
    field_path: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=5, max_length=500)
    suggested_fix: str | None = Field(default=None, max_length=300)


class CriticFeedback(BaseModel):
    """Salida de nivel superior del Critic. El Boss la trata como autoritativa.

    Validador: ``needs_iteration`` exige al menos un issue critical/major. Si el
    Critic solo señala issues minor, debe aceptar — un minor no justifica quemar
    otra llamada al actor.
    """

    verdict: CriticVerdict
    issues: list[CriticIssue] = Field(default_factory=list, max_length=12)
    confidence_in_review: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def iteration_requires_blocking_issue(self) -> CriticFeedback:
        if self.verdict == "needs_iteration":
            blocking = [i for i in self.issues if i.severity in {"critical", "major"}]
            if not blocking:
                raise ValueError(
                    "verdict 'needs_iteration' requires at least one issue with "
                    "severity in {critical, major}; minor-only issues should accept"
                )
        if self.verdict == "reject" and not self.issues:
            raise ValueError("verdict 'reject' requires at least one issue describing why")
        return self
