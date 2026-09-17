import logging

from fastapi import FastAPI

from app.config import APP_NAME, settings
from app.routers import estimations

logging.basicConfig(level=settings.log_level.upper())

app = FastAPI(
    title=APP_NAME,
    description=(
        "API que recibe la transcripción de una reunión y devuelve una "
        "estimación de software generada por un LLM mediante arquitectura CAG "
        "(contexto estático inyectado en el prompt)."
    ),
    version="0.1.0",
)

app.include_router(estimations.router, prefix="/api/v1", tags=["estimations"])


@app.get("/health", tags=["health"], summary="Estado del servicio")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": APP_NAME,
        "env": settings.app_env,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
    }
