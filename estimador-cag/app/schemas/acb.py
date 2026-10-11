"""DTOs para la traza de auditoría del Actor-Critic-Boss.

Viven en ``schemas`` (no en ``services``) para poder referenciarlos tanto desde
el orquestador (``app/services/boss.py``) como desde el modelo de respuesta
(``app/schemas/estimations.py::ACBResponse``) sin ciclos de import.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

BossDecision = Literal["accept", "iterate", "synthesize"]


class ACBIteration(BaseModel):
    """Registro de auditoría de una ronda actor+critic.

    Se persiste en la respuesta para que el llamante pueda mostrar la traza
    completa en la UI (o en un panel de depuración) y para poder razonar sobre
    el sistema de extremo a extremo.
    """

    iteration: int = Field(ge=0)
    decision_after: BossDecision
    critic_verdict: str
    critic_confidence: int = Field(ge=0, le=100)
    issue_summary: list[str] = Field(default_factory=list)


class BossTrace(BaseModel):
    iterations: list[ACBIteration] = Field(default_factory=list)
    final_decision: BossDecision
    iterations_run: int = Field(ge=0)
