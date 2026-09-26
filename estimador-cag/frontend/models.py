"""Modelos de datos del frontend.

Representan el contrato HTTP de la API. Son deliberadamente locales: el
contrato es la API (OpenAPI), no el paquete `app`. Las listas de opciones
duplican los strings de los `Enum` del backend porque el frontend no importa
`app.*`; si cambia el contrato, cambian aquí.
"""

from dataclasses import dataclass

PROJECT_TYPES: list[str] = ["mobile_app", "web_saas", "internal_tool", "data_pipeline"]
DETAIL_LEVELS: list[str] = ["summary", "medium", "detailed"]
OUTPUT_FORMATS: list[str] = ["phases_table", "line_items", "narrative"]


@dataclass(frozen=True)
class ContextResponse:
    system_prompt: str
    description_min_length: int = 0
    description_max_length: int = 0
    llm_configured: bool = False


@dataclass(frozen=True)
class EstimateResponse:
    """Respuesta de `POST /api/v1/estimate`.

    `estimation` es texto libre; `prompt_version` identifica el template que lo
    produjo. El resto son métricas opcionales de la llamada.
    """

    estimation: str
    prompt_version: str
    model: str = ""
    provider: str = ""
    temperature: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    truncated: bool = False
    cache_hit: bool = False
    cost_usd: float | None = None
    fallback_used: bool = False


@dataclass
class StreamMetrics:
    """Métricas de la última generación en streaming (evento `done`)."""

    prompt_version: str = ""
    model: str = ""
    provider: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    truncated: bool = False
    cache_hit: bool = False
    cost_usd: float | None = None
    fallback_used: bool = False
