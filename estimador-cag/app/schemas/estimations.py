"""Contrato HTTP de la API de estimaciones.

Estos modelos son la frontera entre el backend y cualquier cliente (hoy la UI
de Streamlit, mañana otra). La UI no importa este módulo: habla HTTP contra
estos esquemas, que se documentan solos en OpenAPI.

Desde la sesión 04 el request es un formulario tipado: una descripción libre y
tres decisiones de formato cerradas por `Enum`. La respuesta sigue siendo texto
libre (JSON estructurado, guardrails y caché semántico llegan más adelante).
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from app.config import settings


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
    output_format: OutputFormat = Field(
        ..., description="Forma de la estimación renderizada."
    )


class EstimateResponse(BaseModel):
    estimation: str = Field(..., description="Estimación generada en Markdown.")
    prompt_version: str = Field(
        ..., description="Versión del template de prompt que produjo la estimación."
    )
    model: str = Field(..., description="Modelo LLM utilizado.")
    provider: str = Field(..., description="Proveedor LLM utilizado.")
    temperature: float | None = Field(
        None,
        description=(
            "Temperatura usada en la generación. Null para proveedores cuyo "
            "SDK no acepta el parámetro (p. ej. Anthropic)."
        ),
    )
    input_tokens: int | None = Field(None, description="Tokens de entrada consumidos.")
    output_tokens: int | None = Field(None, description="Tokens de salida generados.")
    truncated: bool = Field(
        False,
        description=(
            "True si el modelo se quedó sin presupuesto de salida (LLM_MAX_TOKENS) "
            "y la estimación puede estar incompleta."
        ),
    )
    cache_hit: bool = Field(
        False,
        description="True si la respuesta se sirvió desde la caché sin llamar al proveedor.",
    )
    cost_usd: float | None = Field(
        None,
        description="Coste estimado de la llamada en USD (0 en un acierto de caché).",
    )
    fallback_used: bool = Field(
        False,
        description="True si el modelo primario falló y se usó LLM_FALLBACK_MODEL.",
    )


class ContextResponse(BaseModel):
    system_prompt: str = Field(..., description="System prompt activo renderizado, en Markdown.")
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
