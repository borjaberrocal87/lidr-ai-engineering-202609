import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.config import settings
from app.services.llm_service import (
    LLMConfigurationError,
    LLMInputError,
    LLMProviderError,
    generate_estimation,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

router = APIRouter()


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


@router.post(
    "/estimate",
    response_model=EstimateResponse,
    summary="Genera una estimación de software a partir de una transcripción",
)
def estimate(payload: EstimateRequest) -> EstimateResponse:
    try:
        result = generate_estimation(payload.transcription)
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except LLMInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except LLMProviderError as exc:
        logger.warning("estimate.provider_error", provider=settings.llm_provider)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No se pudo generar la estimación. Inténtalo de nuevo más tarde.",
        ) from exc

    if result.truncated:
        logger.warning("llm.response.truncated", model=result.model, provider=result.provider)

    return EstimateResponse(
        estimation=result.estimation,
        model=result.model,
        provider=result.provider,
        temperature=result.temperature,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        truncated=result.truncated,
    )
