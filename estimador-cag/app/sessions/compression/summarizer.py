"""Resumidor acumulativo para los turnos desalojados.

El resumidor lo invoca ``CompressionPolicy`` cuando turnos que no son ancla se
caen de la ventana deslizante. Recibe el resumen acumulado previo (puede estar
vacío) más los mensajes recién desalojados y los funde en un resumen nuevo. La
salida reemplaza a la anterior — hay un único resumen rodante por sesión, no una
cadena.

Es una llamada LLM aparte. Por defecto usa un modelo barato. Ante un fallo,
mantenemos el resumen previo intacto (mejor perder una pasada de compactación
que borrar estado en el que el modelo ya se apoyaba).
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from app.prompts.loader import render_conversation_summary_prompt
from app.services.llm_wrapper import LLMWrapper
from app.sessions.models import Message

log = structlog.get_logger()

SUMMARY_MAX_TOKENS = 1000
SUMMARY_MAX_RETRIES = 1


class SummaryEnvelope(BaseModel):
    """Envoltorio estructurado mínimo para que el LLM se comprometa con un solo
    campo de texto.

    Usamos Instructor (vía el wrapper) por consistencia con el resto del stack en
    lugar de una completion de texto libre; el prompt instruye al modelo a
    rellenar ``summary`` con el texto del resumen rodante.
    """

    summary: str = Field(min_length=1, max_length=4000)


class CumulativeSummarizer:
    def __init__(self, *, llm_wrapper: LLMWrapper, model: str) -> None:
        self.llm_wrapper = llm_wrapper
        self.model = model

    def summarize(
        self,
        *,
        previous_summary: str | None,
        evicted: list[Message],
    ) -> str:
        """Devuelve el resumen acumulativo actualizado.

        Ante cualquier error del LLM registramos y devolvemos ``previous_summary
        or ""`` para que la sesión siga funcionando. La compresión es best-effort;
        la ventana deslizante garantiza que los turnos recientes siempre estén
        intactos.
        """
        if not evicted:
            return previous_summary or ""

        system_prompt, user_message = render_conversation_summary_prompt(
            previous_summary=previous_summary,
            evicted=evicted,
        )

        try:
            envelope, meta = self.llm_wrapper.complete_structured_messages(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_model=SummaryEnvelope,
                temperature=None,
                max_tokens=SUMMARY_MAX_TOKENS,
                max_retries=SUMMARY_MAX_RETRIES,
                model_override=self.model,
            )
        except Exception as exc:
            log.warning(
                "summarizer_failed",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            return previous_summary or ""

        log.info(
            "summarizer_completed",
            evicted_count=len(evicted),
            previous_chars=len(previous_summary or ""),
            new_chars=len(envelope.summary),
            model=meta.get("model"),
            latency_ms=meta.get("latency_ms"),
        )
        return envelope.summary
