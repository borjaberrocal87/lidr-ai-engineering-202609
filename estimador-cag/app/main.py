import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from starlette.responses import Response

from app.config import APP_NAME, settings
from app.logging_config import configure_logging
from app.routers import estimations, ingestion, sessions

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
