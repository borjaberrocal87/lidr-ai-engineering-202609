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
from uuid import uuid4

from pydantic import BaseModel, Field

from app.domain.schemas.conversation import Message, ProjectMetadata
from app.domain.schemas.observability import TurnObservation


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
    # Sesión 6 — observabilidad y stress test. `turn_count` es un contador
    # monótono (no derivable de `len(history.messages)` porque la compresión
    # desaloja pares) y `last_observation` guarda la foto del último turno para
    # que `GET /sessions/{id}` la exponga sin parsear logs.
    turn_count: int = 0
    last_observation: TurnObservation | None = None
