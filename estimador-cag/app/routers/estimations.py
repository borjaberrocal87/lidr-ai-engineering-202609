from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.llm_service import LLMConfigurationError, generate_estimation

router = APIRouter()


class EstimateRequest(BaseModel):
    transcription: str = Field(
        ...,
        min_length=10,
        description="Texto de la transcripción de la reunión a estimar.",
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
            "Temperatura usada en la generación. Null si el proveedor "
            "(p. ej. Anthropic en este SDK) no la soporta."
        ),
    )
    input_tokens: int | None = Field(None, description="Tokens de entrada consumidos.")
    output_tokens: int | None = Field(None, description="Tokens de salida generados.")


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

    return EstimateResponse(
        estimation=result.estimation,
        model=result.model,
        provider=result.provider,
        temperature=result.temperature,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
