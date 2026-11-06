"""Observabilidad agregada por turno (sesión 6).

Hasta la sesión 5 la estimación conversacional emitía logs sueltos
(``session.estimate_received``, ``llm.structured.end``, ``history_compressed``,
``summarizer_completed``, ``tier_resolved``). Extraer una serie temporal exigía
reconciliar cinco líneas por timestamp.

``TurnObservation`` agrega en un único sitio —y en un único evento structlog—
todo lo que hace falta para medir el CAG turno a turno: tamaño real del prompt
enriquecido, estado de la memoria (ventana, anclas, resumen), consumo del LLM y
tier resuelto. El runner de estrés la lee del snapshot de la sesión
(``GET /api/v1/sessions/{id}``), no de stdout, para que sea determinista y valga
igual en modo in-process que sobre HTTP.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TurnObservation(BaseModel):
    """Foto agregada de un turno conversacional.

    Los trece campos son los mínimos para responder a las tres preguntas del
    baseline: cuánto ocupa el contexto, cuánto cuesta y qué sobrevive de la
    memoria.
    """

    turn_index: int = Field(..., ge=1, description="Turno 1-based dentro de la sesión.")
    session_id: str = Field(..., description="Sesión a la que pertenece el turno.")
    enriched_transcript_chars: int = Field(
        ..., ge=0, description="Caracteres del transcript + adjuntos concatenados."
    )
    attachments_total_chars: int = Field(
        ..., ge=0, description="Caracteres extraídos de los adjuntos (0 si no hay)."
    )
    messages_in_window: int = Field(
        ..., ge=0, description="Mensajes en la ventana deslizante tras la compresión."
    )
    anchors_count: int = Field(..., ge=0, description="Anclas acumuladas en la sesión.")
    summary_chars: int = Field(..., ge=0, description="Caracteres del resumen acumulativo.")
    tokens_in: int | None = Field(None, description="Tokens de entrada de la llamada al LLM.")
    tokens_out: int | None = Field(None, description="Tokens de salida generados.")
    cost_usd: float | None = Field(None, description="Coste estimado de la llamada (USD).")
    latency_ms: float | None = Field(None, description="Latencia de la llamada al LLM (ms).")
    cache_hit_kind: Literal["none", "exact", "semantic"] = Field(
        "none",
        description=(
            "Tipo de acierto de caché. El path conversacional no cachea, así que "
            "siempre es 'none'; el literal existe para reutilizar el tipo en el "
            "path transaccional."
        ),
    )
    last_resolved_tier: str | None = Field(
        None, description="Tier de audiencia resuelto para este turno."
    )

    def to_event(self) -> dict[str, Any]:
        """Payload plano para ``structlog``: una sola línea agregada por turno."""
        return self.model_dump()
