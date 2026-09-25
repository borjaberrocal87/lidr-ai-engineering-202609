from collections.abc import Iterator

import structlog
from fastapi import APIRouter, HTTPException, status
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.config import settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.schemas.estimations import (
    ContextResponse,
    EstimateRequest,
    EstimateResponse,
    EstimationExampleSchema,
    StreamDoneEvent,
)
from app.services.llm_service import (
    LLMConfigurationError,
    LLMInputError,
    LLMProviderError,
    StreamMetrics,
    build_system_prompt,
    generate_estimation,
    stream_estimation,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

router = APIRouter()


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
        cache_hit=result.cache_hit,
        cost_usd=result.cost_usd,
        fallback_used=result.fallback_used,
    )


@router.post(
    "/estimate/stream",
    summary="Genera una estimación en streaming (Server-Sent Events)",
    response_class=EventSourceResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "Flujo SSE. Eventos `token` (deltas de texto), `done` (métricas) "
                "y `error` (fallo de configuración o del proveedor)."
            ),
        },
        422: {"description": "Transcripción fuera de los límites permitidos."},
    },
)
def estimate_stream(payload: EstimateRequest) -> Iterator[ServerSentEvent]:
    """Expone `stream_estimation` como SSE nativo de FastAPI para clientes HTTP.

    Al ser un endpoint generador, la respuesta ya se ha abierto (200) cuando se
    ejecuta el cuerpo, así que cualquier fallo (configuración, proveedor) se
    comunica como evento `error`. La longitud de la transcripción la valida
    Pydantic antes de abrir el flujo (422).
    """
    metrics = StreamMetrics(model="", provider="")
    try:
        tokens = stream_estimation(payload.transcription, metrics)
    except (LLMConfigurationError, LLMInputError) as exc:
        yield ServerSentEvent(event="error", data={"detail": str(exc)})
        return

    try:
        for chunk in tokens:
            yield ServerSentEvent(event="token", data={"text": chunk})
    except LLMProviderError:
        logger.warning("estimate.stream.provider_error", provider=settings.llm_provider)
        yield ServerSentEvent(
            event="error",
            data={"detail": "No se pudo generar la estimación. Inténtalo de nuevo más tarde."},
        )
        return
    except Exception:
        logger.exception("estimate.stream.unexpected_error")
        yield ServerSentEvent(
            event="error",
            data={"detail": "Error inesperado generando la estimación."},
        )
        return

    if metrics.truncated:
        logger.warning("llm.response.truncated", model=metrics.model, provider=metrics.provider)

    done = StreamDoneEvent(
        model=metrics.model,
        provider=metrics.provider,
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        truncated=metrics.truncated,
        cache_hit=metrics.cache_hit,
        cost_usd=metrics.cost_usd,
        fallback_used=metrics.fallback_used,
    )
    yield ServerSentEvent(event="done", data=done.model_dump())


@router.get(
    "/context",
    response_model=ContextResponse,
    summary="Contexto CAG y límites activos (para que la UI no importe el backend)",
)
def context() -> ContextResponse:
    return ContextResponse(
        system_prompt=build_system_prompt(),
        examples=[EstimationExampleSchema(**example) for example in ESTIMATION_EXAMPLES],
        transcription_min_length=settings.transcription_min_length,
        transcription_max_length=settings.transcription_max_length,
        llm_configured=settings.is_configured,
    )
