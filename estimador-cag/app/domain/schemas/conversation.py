"""Contratos compartidos de la conversación.

`Message` y `ProjectMetadata` son vocabulario del dominio: los usan tanto los
prompts (foundation) como el conductor y la capa generativa. Viven aquí para
que `foundation/prompts` no dependa de `generation/conversation`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]


class Message(BaseModel):
    """Un mensaje del historial de conversación."""

    role: Role
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProjectMetadata(BaseModel):
    """Hechos del proyecto que se preservan entre turnos.

    Todos los campos son opcionales: en el primer turno todavía no hay nada
    comprometido. El extractor los rellena turno a turno.
    """

    project_name: str | None = Field(default=None, max_length=120)
    assumed_team_size: int | None = Field(default=None, ge=1, le=50)
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = Field(default=None, max_length=2000)

    def is_empty(self) -> bool:
        """True si no se ha comprometido ningún hecho todavía."""
        return (
            self.project_name is None
            and self.assumed_team_size is None
            and not self.mentioned_technologies
            and self.agreed_scope is None
        )

    def merge_with(self, update: ProjectMetadata) -> ProjectMetadata:
        """Escalares: gana el valor no nulo de `update`. Tecnologías: unión.

        La unión es case-insensitive y preserva el orden original: el usuario
        que añade una tecnología nueva no debe perder las anteriores.
        """
        merged_tech = list(self.mentioned_technologies)
        seen = {tech.lower() for tech in merged_tech}
        for tech in update.mentioned_technologies:
            if tech.lower() not in seen:
                merged_tech.append(tech)
                seen.add(tech.lower())
        return ProjectMetadata(
            project_name=update.project_name or self.project_name,
            assumed_team_size=update.assumed_team_size or self.assumed_team_size,
            mentioned_technologies=merged_tech,
            agreed_scope=update.agreed_scope or self.agreed_scope,
        )
