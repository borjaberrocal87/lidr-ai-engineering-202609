"""Estado en memoria de una sesión conversacional.

Diseño:
- El **system prompt no forma parte del historial**: se regenera en cada turno
  a partir de la `ProjectMetadata` actual (que evoluciona). La invariante es que
  toda llamada al modelo lleva system + ventana de historial + turno actual.
- `ConversationHistory` recorta por pares (user+assistant) para que la
  alternancia de roles no se rompa nunca.
- `ProjectMetadata` mezcla de forma aditiva: los escalares se sobrescriben y la
  lista de tecnologías se une (case-insensitive) para que un stack nuevo no
  borre el anterior.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]


class Message(BaseModel):
    """Un mensaje del historial de conversación."""

    role: Role
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationHistory(BaseModel):
    """Ventana deslizante de turnos (pares user+assistant).

    Cuando se supera `max_turns`, se descartan los pares más antiguos. El system
    prompt no vive aquí: se recibe en `to_messages_list` y se regenera desde la
    metadata actual.
    """

    max_turns: int = Field(default=6, ge=1)
    messages: list[Message] = Field(default_factory=list)

    def append(self, *, user: str, assistant: str) -> None:
        """Añade un turno (user + assistant) y recorta la ventana."""
        self.messages.append(Message(role="user", content=user))
        self.messages.append(Message(role="assistant", content=assistant))
        self._trim()

    def to_messages_list(self, system_prompt: str) -> list[dict[str, str]]:
        """Array `messages` listo para el modelo, con el system regenerado delante.

        El system se recibe por parámetro (no se almacena) porque depende de la
        metadata actual: guardarlo en el historial congelaría una versión vieja.
        """
        return [
            {"role": "system", "content": system_prompt},
            *({"role": message.role, "content": message.content} for message in self.messages),
        ]

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        overflow = len(self.messages) - max_messages
        if overflow > 0:
            if overflow % 2 != 0:  # nunca dejar un par partido
                overflow += 1
            del self.messages[:overflow]


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


class Session(BaseModel):
    """Sesión conversacional. Vive en memoria del proceso.

    Aceptamos la volatilidad en esta fase: reiniciar el servicio borra la
    memoria y, con más de un worker, cada worker tendría su propia copia. La
    persistencia entra en el módulo de RAG.
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    history: ConversationHistory = Field(default_factory=ConversationHistory)
    metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
