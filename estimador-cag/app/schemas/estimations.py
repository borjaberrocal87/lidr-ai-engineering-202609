"""Contrato HTTP de la API de estimaciones.

Estos modelos son la frontera entre el backend y cualquier cliente (hoy la UI
de Streamlit, mañana otra). La UI no importa este módulo: habla HTTP contra
estos esquemas, que se documentan solos en OpenAPI.
"""

from pydantic import BaseModel, Field

from app.config import settings


class EstimateRequest(BaseModel):
    transcription: str = Field(
        ...,
        min_length=settings.transcription_min_length,
        max_length=settings.transcription_max_length,
        description=(
            "Texto de la transcripción de la reunión a estimar. "
            "Longitud acotada por TRANSCRIPTION_MIN_LENGTH y TRANSCRIPTION_MAX_LENGTH."
        ),
        examples=[
            "En la reunión con el equipo de marketing, el cliente explicó que "
            "necesita una landing page con formulario de contacto..."
        ],
    )


class EstimateResponse(BaseModel):
    estimation: str = Field(..., description="Estimación generada en Markdown.")
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


class EstimationExampleSchema(BaseModel):
    meeting_summary: str = Field(..., description="Resumen de la reunión de referencia.")
    estimation: str = Field(..., description="Estimación de referencia generada.")


class ContextResponse(BaseModel):
    system_prompt: str = Field(..., description="System prompt CAG activo, en Markdown.")
    examples: list[EstimationExampleSchema] = Field(
        ..., description="Ejemplos de estimaciones inyectados como contexto (few-shot)."
    )
    transcription_min_length: int = Field(..., description="Longitud mínima aceptada.")
    transcription_max_length: int = Field(..., description="Longitud máxima aceptada.")
    llm_configured: bool = Field(
        ...,
        description=(
            "True si el proveedor activo tiene credenciales. Permite a la UI "
            "avisar sin disparar una llamada al LLM."
        ),
    )


class StreamDoneEvent(BaseModel):
    """Último evento SSE de una estimación en streaming: métricas de la llamada."""

    model: str = Field(..., description="Modelo LLM utilizado.")
    provider: str = Field(..., description="Proveedor LLM utilizado.")
    input_tokens: int | None = Field(None, description="Tokens de entrada consumidos.")
    output_tokens: int | None = Field(None, description="Tokens de salida generados.")
    truncated: bool = Field(False, description="True si el modelo agotó LLM_MAX_TOKENS.")
