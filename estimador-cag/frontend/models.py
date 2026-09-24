"""Modelos de datos del frontend.

Representan la respuesta HTTP de la API. Son deliberadamente locales: el
contrato es la API (OpenAPI), no el paquete `app`.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContextExample:
    meeting_summary: str
    estimation: str


@dataclass(frozen=True)
class ContextResponse:
    system_prompt: str
    examples: list[ContextExample] = field(default_factory=list)
    transcription_min_length: int = 0
    transcription_max_length: int = 0
    llm_configured: bool = False


@dataclass
class StreamMetrics:
    """Métricas de la última generación en streaming (evento `done`)."""

    model: str = ""
    provider: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    truncated: bool = False
