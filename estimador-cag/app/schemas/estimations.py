"""Contrato HTTP de la API de estimaciones.

Estos modelos son la frontera entre el backend y cualquier cliente (hoy la UI
de Streamlit, mañana otra). La UI no importa este módulo: habla HTTP contra
estos esquemas, que se documentan solos en OpenAPI.

Desde la sesión 04 el request es un formulario tipado: una descripción libre y
tres decisiones de formato cerradas por `Enum`. La respuesta sigue siendo texto
libre (JSON estructurado, guardrails y caché semántico llegan más adelante).
"""

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from app.config import settings
from app.sessions.models import ProjectMetadata


class ProjectType(StrEnum):
    """Categoría amplia del proyecto a estimar."""

    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(StrEnum):
    """Profundidad de la estimación."""

    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(StrEnum):
    """Forma de la estimación renderizada."""

    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class ReferenceProject(BaseModel):
    """Proyecto similar que el usuario aporta para calibrar la estimación."""

    name: str = Field(..., min_length=1, max_length=200, description="Nombre del proyecto.")
    description: str = Field(
        ..., min_length=1, max_length=2000, description="Qué era y qué alcance tenía."
    )
    estimation: str | None = Field(
        None,
        max_length=4000,
        description="Resultado de la estimación previa, si se conoce (horas, coste, duración).",
    )


class EstimationRequest(BaseModel):
    """Payload tipado que envía el formulario del cliente."""

    description: str = Field(
        ...,
        min_length=settings.description_min_length,
        max_length=settings.description_max_length,
        description=(
            "Descripción libre del proyecto a estimar. Longitud acotada por "
            "DESCRIPTION_MIN_LENGTH y DESCRIPTION_MAX_LENGTH."
        ),
        examples=[
            "Plataforma web de gestión de inventario para una cadena de 5 tiendas "
            "con control de stock, alertas y dashboard de rotación."
        ],
    )
    project_type: ProjectType = Field(..., description="Categoría amplia del proyecto.")
    detail_level: DetailLevel = Field(..., description="Profundidad de la estimación.")
    output_format: OutputFormat = Field(..., description="Forma de la estimación renderizada.")
    reference_projects: list[ReferenceProject] | None = Field(
        default=None,
        max_length=5,
        description="Proyectos similares para calibrar la estimación (opcional, máx. 5).",
    )


# --- Salida estructurada (sesión 4) -----------------------------------------

OUT_OF_SCOPE_PREFIX = "Out of scope:"
LOW_CONFIDENCE_THRESHOLD = 30


class Phase(BaseModel):
    """Una fase del desglose de la estimación."""

    name: str = Field(..., min_length=1, max_length=64, description="Nombre corto de la fase.")
    duration_weeks: int = Field(..., ge=1, le=52, description="Duración de la fase en semanas.")
    cost_eur: int = Field(..., ge=0, le=1_000_000, description="Coste de la fase en EUR.")
    summary: str = Field(
        ..., min_length=10, max_length=600, description="Qué ocurre en la fase, en una frase."
    )


class EstimationResult(BaseModel):
    """Estimación estructurada y validada por reglas de negocio.

    Los dos validadores son las reglas que el LLM no puede romper: cuando uno
    falla, el wrapper re-promptea al modelo con el mensaje de error.

    El orden de campos es deliberado: ``phases`` va **antes** que los totales
    para que el modelo se comprometa primero con las cifras por fase (generación
    autorregresiva) y solo después las sume. Al revés, tiende a elegir un total
    redondo y a ajustar las fases a la fuerza, lo que hace mal.
    """

    summary: str = Field(..., min_length=10, max_length=1200, description="Resumen ejecutivo.")
    confidence_pct: int = Field(..., ge=0, le=100, description="Confianza de la estimación (%).")
    phases: list[Phase] = Field(..., min_length=1, max_length=8, description="Desglose por fases.")
    total_duration_weeks: int = Field(..., ge=1, le=104, description="Duración total en semanas.")
    total_cost_eur: int = Field(..., ge=0, le=2_000_000, description="Coste total en EUR.")

    @model_validator(mode="after")
    def phases_sum_matches_total(self) -> "EstimationResult":
        phase_sum = sum(phase.cost_eur for phase in self.phases)
        if phase_sum != self.total_cost_eur:
            raise ValueError(
                f"la suma de las fases ({phase_sum} EUR) no coincide con total_cost_eur "
                f"({self.total_cost_eur} EUR); ajusta las fases o el total"
            )
        return self

    @model_validator(mode="after")
    def low_confidence_requires_out_of_scope_prefix(self) -> "EstimationResult":
        if self.confidence_pct < LOW_CONFIDENCE_THRESHOLD and not self.summary.startswith(
            OUT_OF_SCOPE_PREFIX
        ):
            raise ValueError(
                f"confidence_pct < {LOW_CONFIDENCE_THRESHOLD} exige que el summary empiece "
                f"por {OUT_OF_SCOPE_PREFIX!r}; rechaza la estimación si la descripción es "
                f"demasiado vaga para dimensionarla"
            )
        return self

    @property
    def out_of_scope(self) -> bool:
        """True si el modelo declaró baja confianza (fuera de alcance)."""
        return self.confidence_pct < LOW_CONFIDENCE_THRESHOLD


class StructuredEstimateResponse(BaseModel):
    """Respuesta de `POST /api/v1/estimate` a partir de la sesión 4."""

    result: EstimationResult = Field(..., description="Estimación estructurada y validada.")
    prompt_version: str = Field(..., description="Versión del template de prompt usada.")
    cached: bool = Field(False, description="True si vino de la caché (exacta o semántica).")
    model: str = Field("", description="Modelo LLM utilizado.")
    provider: str = Field("", description="Proveedor LLM utilizado.")
    input_tokens: int | None = Field(None, description="Tokens de entrada consumidos.")
    output_tokens: int | None = Field(None, description="Tokens de salida generados.")
    cost_usd: float | None = Field(None, description="Coste estimado de la llamada (USD).")
    fallback_used: bool = Field(False, description="True si se usó el modelo de fallback.")


class SessionEstimateResponse(BaseModel):
    """Respuesta de `POST /api/v1/sessions/{session_id}/estimate` (sesión 5).

    Añade al resultado estructurado la metadata de la sesión (ya actualizada
    con este turno) y el tamaño del historial, para que el cliente pueda pintar
    el panel de memoria sin una segunda llamada.
    """

    session_id: str = Field(..., description="Sesión a la que pertenece el turno.")
    result: EstimationResult = Field(..., description="Estimación estructurada y validada.")
    prompt_version: str = Field(..., description="Versión del template de prompt usada.")
    metadata: ProjectMetadata = Field(..., description="Hechos del proyecto tras el turno.")
    history_messages: int = Field(..., description="Mensajes en el historial efectivo.")
    model: str = Field("", description="Modelo LLM utilizado.")
    provider: str = Field("", description="Proveedor LLM utilizado.")
    input_tokens: int | None = Field(None, description="Tokens de entrada consumidos.")
    output_tokens: int | None = Field(None, description="Tokens de salida generados.")
    cost_usd: float | None = Field(None, description="Coste estimado de la llamada (USD).")
    fallback_used: bool = Field(False, description="True si se usó el modelo de fallback.")


class ContextResponse(BaseModel):
    system_prompt: str = Field(..., description="System prompt activo renderizado, en Markdown.")
    prompt_version: str = Field(..., description="Versión del prompt renderizado.")
    available_versions: list[str] = Field(
        ..., description="Versiones de prompt disponibles en el servicio."
    )
    description_min_length: int = Field(..., description="Longitud mínima aceptada.")
    description_max_length: int = Field(..., description="Longitud máxima aceptada.")
    llm_configured: bool = Field(
        ...,
        description=(
            "True si el proveedor activo tiene credenciales. Permite a la UI "
            "avisar sin disparar una llamada al LLM."
        ),
    )


class StreamDoneEvent(BaseModel):
    """Último evento SSE de una estimación en streaming: métricas de la llamada."""

    prompt_version: str = Field(..., description="Versión del template de prompt utilizada.")
    model: str = Field(..., description="Modelo LLM utilizado.")
    provider: str = Field(..., description="Proveedor LLM utilizado.")
    input_tokens: int | None = Field(None, description="Tokens de entrada consumidos.")
    output_tokens: int | None = Field(None, description="Tokens de salida generados.")
    truncated: bool = Field(False, description="True si el modelo agotó LLM_MAX_TOKENS.")
    cache_hit: bool = Field(False, description="True si la respuesta vino de la caché.")
    cost_usd: float | None = Field(None, description="Coste estimado de la llamada (USD).")
    fallback_used: bool = Field(False, description="True si se usó el modelo de fallback.")
