from collections.abc import Iterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.config import settings
from app.prompts.loader import (
    DEFAULT_ESTIMATION_PROMPT_VERSION,
    available_estimation_versions,
    render_estimation_prompt,
)
from app.schemas.estimations import (
    ContextResponse,
    DetailLevel,
    EstimateResponse,
    EstimationRequest,
    OutputFormat,
    ProjectType,
    StreamDoneEvent,
)
from app.services.llm_service import (
    LLMConfigurationError,
    LLMInputError,
    LLMProviderError,
    StreamMetrics,
    generate_estimation,
    stream_estimation,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

router = APIRouter()

PROMPT_VERSION = DEFAULT_ESTIMATION_PROMPT_VERSION


def _default_estimation_request() -> EstimationRequest:
    """Request neutro para inspeccionar el prompt renderizado en `/context`."""
    return EstimationRequest(
        description="Proyecto de ejemplo para inspeccionar el prompt de estimación activo.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )


def _resolve_prompt_version(version: str) -> str:
    """Valida la versión pedida contra las disponibles en disco (404 si no existe)."""
    available = available_estimation_versions()
    if version not in available:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Versión de prompt desconocida: {version!r}. "
                f"Disponibles: {', '.join(available) or 'ninguna'}."
            ),
        )
    return version


def validated_prompt_version(prompt_version: str = PROMPT_VERSION) -> str:
    """Dependencia que valida el query param `prompt_version` antes del handler.

    Al resolverla FastAPI antes de ejecutar el endpoint, el 404 se devuelve antes
    de abrir el flujo SSE (que ya no podría cambiar el estado).
    """
    return _resolve_prompt_version(prompt_version)


@router.post(
    "/estimate",
    response_model=EstimateResponse,
    summary="Genera una estimación de software a partir de un formulario tipado",
)
def estimate(
    payload: EstimationRequest,
    version: str = Depends(validated_prompt_version),
) -> EstimateResponse:
    try:
        result = generate_estimation(payload, version=version)
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

    logger.info(
        "estimate.request",
        prompt_version=version,
        project_type=payload.project_type.value,
        detail_level=payload.detail_level.value,
        output_format=payload.output_format.value,
        description_chars=len(payload.description),
    )

    return EstimateResponse(
        estimation=result.estimation,
        prompt_version=version,
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
        422: {"description": "Descripción fuera de los límites permitidos."},
    },
)
def estimate_stream(
    payload: EstimationRequest,
    version: str = Depends(validated_prompt_version),
) -> Iterator[ServerSentEvent]:
    """Expone `stream_estimation` como SSE nativo de FastAPI para clientes HTTP.

    Al ser un endpoint generador, la respuesta ya se ha abierto (200) cuando se
    ejecuta el cuerpo, así que cualquier fallo (configuración, proveedor) se
    comunica como evento `error`. El request y la versión de prompt se validan
    antes de abrir el flujo (422 y 404) mediante Pydantic y la dependencia.
    """
    yield from _stream_estimation_events(payload, version)


def _stream_estimation_events(
    payload: EstimationRequest,
    version: str,
) -> Iterator[ServerSentEvent]:
    metrics = StreamMetrics(model="", provider="")
    try:
        tokens = stream_estimation(payload, metrics, version=version)
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
        prompt_version=version,
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
    summary="Prompt activo y límites (para que la UI no importe el backend)",
)
def context(prompt_version: str = PROMPT_VERSION) -> ContextResponse:
    version = _resolve_prompt_version(prompt_version)
    default_request = _default_estimation_request()
    system_prompt, _ = render_estimation_prompt(default_request, version=version)
    return ContextResponse(
        system_prompt=system_prompt,
        prompt_version=version,
        available_versions=available_estimation_versions(),
        description_min_length=settings.description_min_length,
        description_max_length=settings.description_max_length,
        llm_configured=settings.is_configured,
    )
