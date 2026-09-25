import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import FastAPI, Request
from starlette.responses import Response

from app.config import APP_NAME, settings
from app.logging_config import configure_logging
from app.routers import estimations

configure_logging()

app = FastAPI(
    title=APP_NAME,
    description=(
        "API que recibe la transcripción de una reunión y devuelve una "
        "estimación de software generada por un LLM mediante arquitectura CAG "
        "(contexto estático inyectado en el prompt)."
    ),
    version="0.1.0",
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
