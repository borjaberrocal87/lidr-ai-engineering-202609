"""Modelos de datos del frontend.

Representan el contrato HTTP de la API. Son deliberadamente locales: el
contrato es la API (OpenAPI), no el paquete `app`. Las listas de opciones
duplican los strings de los `Enum` del backend porque el frontend no importa
`app.*`; si cambia el contrato, cambian aquí.
"""

from dataclasses import dataclass, field

PROJECT_TYPES: list[str] = ["mobile_app", "web_saas", "internal_tool", "data_pipeline"]
DETAIL_LEVELS: list[str] = ["summary", "medium", "detailed"]
OUTPUT_FORMATS: list[str] = ["phases_table", "line_items", "narrative"]


@dataclass(frozen=True)
class ContextResponse:
    system_prompt: str
    prompt_version: str = ""
    available_versions: list[str] = field(default_factory=list)
    description_min_length: int = 0
    description_max_length: int = 0
    llm_configured: bool = False


@dataclass(frozen=True)
class Phase:
    """Una fase del desglose devuelto por el backend."""

    name: str
    duration_weeks: int
    cost_eur: int
    summary: str


@dataclass(frozen=True)
class EstimationResult:
    """Estimación estructurada y validada (contrato de la sesión 4)."""

    summary: str
    confidence_pct: int
    phases: list[Phase]
    total_duration_weeks: int
    total_cost_eur: int

    @property
    def out_of_scope(self) -> bool:
        return self.confidence_pct < 30


@dataclass(frozen=True)
class StructuredEstimateResponse:
    """Respuesta de `POST /api/v1/estimate` (contrato estructurado)."""

    result: EstimationResult
    prompt_version: str
    cached: bool = False
    model: str = ""
    provider: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    fallback_used: bool = False


@dataclass(frozen=True)
class ProjectMetadata:
    """Hechos del proyecto que el backend preserva entre turnos (sesión 5)."""

    project_name: str | None = None
    assumed_team_size: int | None = None
    mentioned_technologies: list[str] = field(default_factory=list)
    agreed_scope: str | None = None


@dataclass(frozen=True)
class SessionInfo:
    """Respuesta de `GET /api/v1/sessions/{session_id}`."""

    session_id: str
    message_count: int
    max_turns: int
    metadata: ProjectMetadata


@dataclass(frozen=True)
class SessionEstimateResponse:
    """Respuesta de `POST /api/v1/sessions/{session_id}/estimate` (sesión 5)."""

    session_id: str
    result: EstimationResult
    prompt_version: str
    metadata: ProjectMetadata
    history_messages: int
    model: str = ""
    provider: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
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
