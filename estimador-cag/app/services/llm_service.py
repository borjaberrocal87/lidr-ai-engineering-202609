"""Servicio de estimación con prompts versionados.

Este módulo se queda con la lógica de negocio: validar la entrada, renderizar el
par de prompts desde el loader de plantillas Jinja2 y mapear la respuesta al
contrato del servicio. La llamada al proveedor (fallback, caché, coste, logging)
vive en `app.services.llm_wrapper`.

El contenido del prompt ya no es código: viene de
`app/prompts/estimation/<version>/`, de modo que cambiar el texto, los ejemplos
o las reglas no obliga a tocar el servicio.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import structlog

from app.config import DEFAULT_TEMPERATURE, reveal_secret, settings
from app.dependencies import get_llm_wrapper
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimations import EstimationRequest
from app.services.errors import (
    LLMConfigurationError as LLMConfigurationError,
)
from app.services.errors import (
    LLMInputError as LLMInputError,
)
from app.services.errors import (
    LLMProviderError as LLMProviderError,
)
from app.services.llm_wrapper import StreamMetrics as StreamMetrics

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


@dataclass
class EstimationResult:
    estimation: str
    model: str
    provider: str
    temperature: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    truncated: bool = False
    cache_hit: bool = False
    cost_usd: float | None = None
    fallback_used: bool = False


def _check_description(description: str) -> None:
    """Guard de longitud en el servicio, por si el llamante no es la API.

    La API y Streamlit ya validan en el borde; esto evita que un worker, un CLI
    o una cola manden texto sin techo al proveedor.
    """
    min_length = settings.description_min_length
    max_length = settings.description_max_length
    if not min_length <= len(description) <= max_length:
        raise LLMInputError(
            f"La descripción debe tener entre {min_length} y {max_length} caracteres "
            f"(tiene {len(description)})."
        )


def _require_configuration() -> None:
    """Valida de forma anticipada que el proveedor activo tiene credenciales."""
    provider = settings.llm_provider
    if provider == "openai":
        if not reveal_secret(settings.open_ai_key):
            raise LLMConfigurationError("OPEN_AI_KEY no está configurada. Añádela al archivo .env.")
    elif provider == "anthropic":
        if not reveal_secret(settings.anthropic_api_key):
            raise LLMConfigurationError(
                "ANTHROPIC_API_KEY no está configurada. Añádela al archivo .env."
            )
    elif provider == "custom":
        if not settings.custom_llm_base_url:
            raise LLMConfigurationError(
                "CUSTOM_LLM_BASE_URL no está configurada. Añádela al archivo .env."
            )
        if not reveal_secret(settings.custom_llm_api_key):
            raise LLMConfigurationError(
                "CUSTOM_LLM_API_KEY no está configurada. Añádela al archivo .env."
            )
    else:
        raise LLMConfigurationError(
            f"Proveedor LLM no soportado: {provider!r}. "
            "Usa 'openai', 'anthropic' o 'custom' en LLM_PROVIDER."
        )


_temperature_warning_emitted = False


def _warn_if_temperature_ignored() -> None:
    """Avisa una sola vez si se configura una temperatura que Anthropic ignora.

    El SDK de Anthropic (1.5.0) no acepta el parámetro `temperature`, así que la
    variable `TEMPERATURE` es configuración muerta para ese proveedor.
    """
    global _temperature_warning_emitted
    if _temperature_warning_emitted:
        return
    if settings.llm_provider == "anthropic" and settings.temperature != DEFAULT_TEMPERATURE:
        logger.warning(
            "llm.temperature.ignored",
            provider="anthropic",
            temperature=settings.temperature,
        )
        _temperature_warning_emitted = True


def _temperature_for_result() -> float | None:
    """Temperatura efectiva: Anthropic no la admite, el resto sí."""
    if settings.llm_provider == "anthropic":
        return None
    return settings.temperature


def generate_estimation(request: EstimationRequest) -> EstimationResult:
    """Genera una estimación a partir de un request tipado usando el proveedor configurado."""
    _check_description(request.description)
    _require_configuration()
    _warn_if_temperature_ignored()

    system_prompt, user_message = render_estimation_prompt(request)
    result = get_llm_wrapper().complete(
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=settings.temperature,
        max_tokens=settings.llm_max_tokens,
        cache_user_message=request.description,
    )

    text = result["estimation"]
    if not text:
        raise LLMProviderError(
            f"El proveedor LLM '{result['provider']}' devolvió una respuesta vacía."
        )

    return EstimationResult(
        estimation=text,
        model=result["model"],
        provider=result["provider"],
        temperature=_temperature_for_result(),
        input_tokens=result.get("input_tokens"),
        output_tokens=result.get("output_tokens"),
        truncated=bool(result.get("truncated", False)),
        cache_hit=bool(result.get("cache_hit", False)),
        cost_usd=result.get("cost_usd"),
        fallback_used=bool(result.get("fallback_used", False)),
    )


def _ensure_non_empty(inner: Iterator[str]) -> Iterator[str]:
    """Convierte un generador vacío en un error de proveedor."""
    produced = False
    for chunk in inner:
        produced = True
        yield chunk
    if not produced:
        raise LLMProviderError("El proveedor LLM devolvió una respuesta vacía.")


def stream_estimation(
    request: EstimationRequest,
    metrics: StreamMetrics | None = None,
) -> Iterator[str]:
    """Genera una estimación en streaming, cediendo el texto token a token.

    Valida la configuración de forma anticipada (al llamar a la función), de modo
    que los errores de configuración se propagan antes de empezar a consumir el
    generador. Si se pasa `metrics`, se rellena con el modelo, los tokens, el
    coste, si hubo acierto de caché y si se usó el fallback.
    """
    _check_description(request.description)
    _require_configuration()
    _warn_if_temperature_ignored()

    system_prompt, user_message = render_estimation_prompt(request)
    active_metrics = metrics if metrics is not None else StreamMetrics()
    inner = get_llm_wrapper().complete_stream(
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=settings.temperature,
        max_tokens=settings.llm_max_tokens,
        metrics=active_metrics,
        cache_user_message=request.description,
    )
    return _ensure_non_empty(inner)
