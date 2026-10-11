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
    """Ventana deslizante de turnos (pares user+assistant) con memoria híbrida.

    Tres huecos de almacenamiento (sesión 5, directo):

    - ``messages``: la ventana deslizante reciente (últimos ``max_turns`` pares).
    - ``anchors``: turnos marcados por ``AnchorDetector`` como compromisos
      durables (NDA firmado, alcance congelado, contexto de compliance…). Viven
      fuera de la ventana y nunca se desalojan.
    - ``summary``: resumen acumulativo en texto libre de turnos antiguos que ya
      plegó la ``CompressionPolicy``.

    ``to_messages()`` compone los tres en el array que ve el LLM:

        [sobre de resumen?] + anclas_en_orden + ventana_reciente

    El recorte ya NO ocurre en ``append``: lo gestiona ``CompressionPolicy``, que
    es la única fuente de verdad sobre qué olvidar.
    """

    max_turns: int = Field(default=6, ge=1)
    messages: list[Message] = Field(default_factory=list)
    anchors: list[Message] = Field(default_factory=list)
    summary: str | None = Field(default=None)

    def append(self, *, user: str, assistant: str) -> None:
        """Añade un turno (user + assistant).

        La compresión (promoción de anclas + resumen acumulativo + ventana) NO se
        aplica aquí — es trabajo de ``CompressionPolicy.apply`` (y su wrapper
        ``apply_compression``). El servicio es responsable de invocarla tras cada
        turno. Mantener la estructura de datos tonta hace que la política sea la
        única fuente de verdad de la ventana, las anclas y el resumen.
        """
        self.messages.append(Message(role="user", content=user))
        self.messages.append(Message(role="assistant", content=assistant))

    def to_messages(self) -> list[dict[str, str]]:
        """Array `messages` (sin system) con resumen + anclas + ventana reciente.

        El orden es intencionado: el resumen ancla al modelo en la conversación
        temprana, las anclas portan los compromisos irrenunciables literal, y la
        ventana reciente aporta el contexto turno a turno activo.
        """
        out: list[dict[str, str]] = []
        if self.summary:
            # Se envuelve como mensaje de usuario sintético para que los
            # proveedores lo enruten limpio a través del scaffolding de tool-use.
            out.append(
                {
                    "role": "user",
                    "content": (
                        "[Earlier conversation summary — the recent turns "
                        "below are the live thread]\n" + self.summary
                    ),
                }
            )
        for anchor in self.anchors:
            out.append({"role": anchor.role, "content": anchor.content})
        for message in self.messages:
            out.append({"role": message.role, "content": message.content})
        return out

    def to_messages_list(self, system_prompt: str) -> list[dict[str, str]]:
        """Array `messages` listo para el modelo, con el system regenerado delante.

        El system se recibe por parámetro (no se almacena) porque depende de la
        metadata actual: guardarlo en el historial congelaría una versión vieja.
        """
        return [{"role": "system", "content": system_prompt}, *self.to_messages()]


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
    # Caché del último tier resuelto, para que el panel lateral pueda mostrarlo
    # sin re-ejecutar el resolver (patrón Actor-Critic-Boss / tier dinámico).
    last_resolved_tier: str | None = None
    last_tier_rule: str | None = None
