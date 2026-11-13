import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from slowapi.errors import RateLimitExceeded
from starlette.responses import Response

from app.api import config as config_router
from app.api import (
    estimate_tasks,
    estimations,
    ingestion,
    rag_estimation,
    retrieval,
    retrieval_multi_source,
    search,
    sessions,
)
from app.api.embeddings import router as embeddings_router
from app.api.rate_limiting import limiter, rate_limit_handler
from app.config import APP_NAME, settings
from app.foundation.observability.logging import configure_logging

configure_logging()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Arranque/parada. Prevalida el catálogo de ingesta (fail-fast).

    Un catálogo mal formado es un problema de despliegue, no de request: mejor
    detectarlo al arrancar que en la primera llamada a `/ingestion/runs`.
    """
    log = structlog.get_logger()
    try:
        from app.dependencies import get_catalog

        catalog = get_catalog()
        log.info(
            "catalog_loaded",
            version=catalog.version,
            sources_total=len(catalog.sources),
            sources_included=len(catalog.included_sources()),
        )
    except Exception as exc:
        log.error("catalog_load_failed", error=str(exc)[:400])
    yield
    log.info("application_shutdown")


app = FastAPI(
    title=APP_NAME,
    description=(
        "API que recibe la transcripción de una reunión y devuelve una "
        "estimación de software generada por un LLM mediante arquitectura CAG "
        "(contexto estático inyectado en el prompt)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Propaga un `request_id` correlacionable por todas las capas."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    # `state` permite que cualquier capa (p. ej. el orquestador RAG) lea el id
    # sin depender del contextvars de structlog.
    request.state.request_id = request_id
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.clear_contextvars()
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(estimations.router, prefix="/api/v1", tags=["estimations"])
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(ingestion.router, prefix="/api/v1")
app.include_router(embeddings_router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(config_router.router, prefix="/api/v1")
# Sesión 9 — RAG end-to-end.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
app.include_router(retrieval.router, prefix="/api/v1")
app.include_router(retrieval_multi_source.router, prefix="/api/v1")
app.include_router(rag_estimation.router, prefix="/api/v1")
app.include_router(estimate_tasks.router, prefix="/api/v1")


@app.get("/health", tags=["health"], summary="Estado del servicio")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "service": APP_NAME,
        "env": settings.app_env,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "routing_mode": settings.llm_routing_mode,
        "llm_configured": settings.is_configured,
    }
