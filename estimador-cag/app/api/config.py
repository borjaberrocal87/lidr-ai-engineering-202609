"""Capa HTTP de la configuración de modelos en runtime (Ajustes).

Router fino: la validación de la actualización parcial es la única lógica; el
store de overrides vive en ``app/foundation/llm/runtime_config.py``. Un cambio de
modelo aplica en la SIGUIENTE llamada LLM (wrapper y servicio leen el store por
llamada); no se reconstruye nada ni hace falta reiniciar.
"""

from __future__ import annotations

from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings, reveal_secret
from app.dependencies import get_runtime_config, get_runtime_retrieval_config
from app.foundation.llm.runtime_config import (
    FRESHNESS_KEY,
    MODEL_KEYS,
    QUERY_PLAN_KEY,
    ROUTING_KEY,
    RuntimeConfigUnavailable,
    RuntimeModelConfig,
    RuntimeRetrievalConfig,
)
from app.foundation.llm.wrapper import provider_from_model

log = structlog.get_logger()

# Mapea el campo de la petición PUT → la clave del hash de Redis.
_STAGE_TOGGLE_KEYS = {
    "routing_enabled": ROUTING_KEY,
    "query_plan_enabled": QUERY_PLAN_KEY,
    "freshness_enabled": FRESHNESS_KEY,
}

router = APIRouter(prefix="/config", tags=["config"])

EMBEDDING_MODEL_NOTE = "Solo lectura: cambiarlo invalidaría todos los vectores almacenados."

PROVIDER_KEY_FIELDS = {
    "openai": "OPEN_AI_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "custom": "CUSTOM_LLM_BASE_URL + CUSTOM_LLM_API_KEY",
}


class ModelUpdateRequest(BaseModel):
    """Actualización parcial: solo se tocan las claves presentes; ``null`` resetea."""

    models: dict[str, str | None] = Field(min_length=1)


class RetrievalUpdateRequest(BaseModel):
    """Partial update of the Session 10 retrieval toggles.

    Only the keys present are touched; explicit ``null`` resets a knob to its
    .env default.
    """

    search_mode: str | None = Field(default=None, description="'vector' or 'hybrid'.")
    rerank: bool | None = Field(default=None, description="Enable cross-encoder reranking.")
    # Sesión 10 — etapas de la recuperación multifuente.
    routing_enabled: bool | None = Field(
        default=None, description="Enable collection routing."
    )
    query_plan_enabled: bool | None = Field(
        default=None, description="Enable query expansion/decomposition."
    )
    freshness_enabled: bool | None = Field(
        default=None, description="Enable temporal freshness re-weighting."
    )
    # Sesión 10 — knobs numéricos del consenso de horas por tarea.
    task_hours_top_k: int | None = Field(
        default=None, ge=1, le=30, description="Neighbours per task for the hours consensus."
    )
    task_hours_distance_threshold: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Red-flag floor: no match beyond this distance."
    )


def _provider_configured(settings: Settings, provider: str) -> bool:
    if provider == "openai":
        return bool(reveal_secret(settings.open_ai_key))
    if provider == "anthropic":
        return bool(reveal_secret(settings.anthropic_api_key))
    if provider == "custom":
        return bool(settings.custom_llm_base_url) and bool(
            reveal_secret(settings.custom_llm_api_key)
        )
    return False


def _available_models(settings: Settings) -> list[str]:
    """Catálogo filtrado por las API keys realmente configuradas."""
    available: list[str] = []
    for model in settings.available_models:
        provider = provider_from_model(model)
        if _provider_configured(settings, provider):
            available.append(model)
    return available


def _config_payload(runtime_config: RuntimeModelConfig, settings: Settings) -> dict[str, Any]:
    return {
        "models": runtime_config.snapshot(),
        "available_models": _available_models(settings),
        "embedding_model": settings.embedding_model,
        "embedding_model_note": EMBEDDING_MODEL_NOTE,
    }


@router.get("/models")
def get_models(
    runtime_config: Annotated[RuntimeModelConfig, Depends(get_runtime_config)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Configuración actual: effective/default/overridden por knob."""
    return _config_payload(runtime_config, settings)


@router.put("/models")
def update_models(
    request: ModelUpdateRequest,
    runtime_config: Annotated[RuntimeModelConfig, Depends(get_runtime_config)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Aplica una actualización parcial de overrides. Todo o nada: se valida
    cada entrada ANTES de escribir nada."""
    available = _available_models(settings)

    for key, value in request.models.items():
        if key not in MODEL_KEYS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Clave de modelo desconocida: {key}",
            )
        if value is None:
            continue
        if value not in settings.available_models:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"El modelo '{value}' no está en el catálogo",
            )
        if value not in available:
            key_field = PROVIDER_KEY_FIELDS.get(provider_from_model(value), "API key")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"El modelo '{value}' requiere {key_field}, que no está configurada",
            )

    try:
        for key, value in request.models.items():
            old_effective = runtime_config.effective(key)
            runtime_config.set(key, value)
            log.info(
                "runtime_config_changed",
                key=key,
                old_effective=old_effective,
                new_value=value,
                reset=value is None,
            )
    except RuntimeConfigUnavailable as exc:
        log.error("runtime_config_write_failed", error=str(exc)[:200])
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El store de configuración runtime no está disponible.",
        ) from exc

    return _config_payload(runtime_config, settings)


@router.get("/retrieval")
def get_retrieval(
    runtime_retrieval: Annotated[RuntimeRetrievalConfig, Depends(get_runtime_retrieval_config)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Current retrieval configuration: effective/default/overridden per toggle."""
    return {
        "retrieval": runtime_retrieval.snapshot(),
        "reranker_model": settings.reranker_model,
    }


@router.put("/retrieval")
def update_retrieval(
    request: RetrievalUpdateRequest,
    runtime_retrieval: Annotated[RuntimeRetrievalConfig, Depends(get_runtime_retrieval_config)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Flip search mode and/or reranking at runtime — effective on the next call."""
    sent = request.model_fields_set
    try:
        if "search_mode" in sent:
            try:
                runtime_retrieval.set_search_mode(request.search_mode)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc
            log.info(
                "runtime_retrieval_changed",
                key="search_mode",
                new_value=request.search_mode,
            )
        if "rerank" in sent:
            runtime_retrieval.set_rerank(request.rerank)
            log.info("runtime_retrieval_changed", key="rerank", new_value=request.rerank)
        for field_name, hash_key in _STAGE_TOGGLE_KEYS.items():
            if field_name in sent:
                runtime_retrieval.set_bool(hash_key, getattr(request, field_name))
                log.info(
                    "runtime_retrieval_changed",
                    key=field_name,
                    new_value=getattr(request, field_name),
                )
        if "task_hours_top_k" in sent:
            try:
                runtime_retrieval.set_task_hours_top_k(request.task_hours_top_k)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc
            log.info(
                "runtime_retrieval_changed",
                key="task_hours_top_k",
                new_value=request.task_hours_top_k,
            )
        if "task_hours_distance_threshold" in sent:
            try:
                runtime_retrieval.set_task_hours_distance_threshold(
                    request.task_hours_distance_threshold
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc
            log.info(
                "runtime_retrieval_changed",
                key="task_hours_distance_threshold",
                new_value=request.task_hours_distance_threshold,
            )
    except RuntimeConfigUnavailable as exc:
        log.error("runtime_retrieval_write_failed", error=str(exc)[:200])
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Runtime config store unavailable.",
        ) from exc

    return {
        "retrieval": runtime_retrieval.snapshot(),
        "reranker_model": settings.reranker_model,
    }
