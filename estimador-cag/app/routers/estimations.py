import json
from collections.abc import Iterator
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

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
    )


@router.post(
    "/estimate/stream",
    summary="Genera una estimación en streaming (Server-Sent Events)",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "Flujo SSE. Eventos `token` (deltas de texto), `done` (métricas) "
                "y `error` (fallo del proveedor a mitad del stream)."
            ),
        },
        422: {"description": "Transcripción fuera de los límites permitidos."},
        503: {"description": "El proveedor activo no está configurado."},
    },
)
def estimate_stream(payload: EstimateRequest) -> StreamingResponse:
    """Expone `stream_estimation` como SSE para clientes HTTP (UI, curl, etc.).

    La configuración del proveedor se valida al invocar `stream_estimation`, de
    modo que los errores de configuración/entrada se devuelven como HTTP normal
    (503/422) antes de abrir el stream. Los fallos del proveedor que ocurren al
    consumir el generador se emiten como un evento SSE `error`.
    """
    metrics = StreamMetrics(model="", provider="")
    try:
        tokens = stream_estimation(payload.transcription, metrics)
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

    return StreamingResponse(
        _sse_stream(tokens, metrics),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_stream(tokens: Iterator[str], metrics: StreamMetrics) -> Iterator[str]:
    """Traduce el generador del servicio a eventos SSE.

    `stream_estimation` ya ha validado config/entrada antes de llegar aquí; lo
    que puede fallar es el proveedor durante la iteración, y eso se comunica a
    la UI como evento `error` porque la respuesta HTTP ya está abierta (200).
    """
    try:
        for chunk in tokens:
            yield _sse("token", {"text": chunk})
    except LLMProviderError:
        logger.warning("estimate.stream.provider_error", provider=settings.llm_provider)
        yield _sse(
            "error",
            {"detail": "No se pudo generar la estimación. Inténtalo de nuevo más tarde."},
        )
        return
    except Exception:
        logger.exception("estimate.stream.unexpected_error")
        yield _sse("error", {"detail": "Error inesperado generando la estimación."})
        return

    if metrics.truncated:
        logger.warning("llm.response.truncated", model=metrics.model, provider=metrics.provider)

    done = StreamDoneEvent(
        model=metrics.model,
        provider=metrics.provider,
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        truncated=metrics.truncated,
    )
    yield _sse("done", done.model_dump())
