"""Camino de texto libre y streaming (legado de la sesión 3).

Este módulo alberga el contrato de **texto libre** —token a token— que quedó al
margen de la salida estructurada de la sesión 4. Vive separado del orquestador
``EstimationService`` porque su forma de respuesta es distinta (un generador de
strings y eventos SSE, no un objeto JSON validado).

Contiene:

- ``generate_estimation``: generación bloqueante de texto libre.
- ``stream_estimation``: generación en streaming (usada por ``/estimate/stream``).

Los helpers de validación de entrada y de clave de caché se comparten con
``app.services.estimation`` para no duplicar comportamiento.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from app.config import settings
from app.dependencies import get_llm_wrapper
from app.prompts.loader import DEFAULT_ESTIMATION_PROMPT_VERSION, render_estimation_prompt
from app.schemas.estimations import EstimationRequest
from app.services.errors import LLMProviderError
from app.services.estimation import (
    cache_key_material,
    check_description,
    require_configuration,
    run_input_guardrails,
    temperature_for_result,
    warn_if_temperature_ignored,
)
from app.services.llm_wrapper import StreamMetrics as StreamMetrics


@dataclass
class TextEstimationResult:
    """Resultado de texto libre (legado de la sesión 3, se mantiene para `/stream`)."""

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


def generate_estimation(
    request: EstimationRequest,
    version: str = DEFAULT_ESTIMATION_PROMPT_VERSION,
) -> TextEstimationResult:
    """Genera una estimación de **texto libre** (legado) usando el proveedor configurado."""
    check_description(request.description)
    require_configuration()
    run_input_guardrails(request.description)
    warn_if_temperature_ignored()

    system_prompt, user_message = render_estimation_prompt(request, version=version)
    result = get_llm_wrapper().complete(
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=settings.temperature,
        max_tokens=settings.llm_max_tokens,
        cache_key=cache_key_material(request, version),
    )

    text = result["estimation"]
    if not text:
        raise LLMProviderError(
            f"El proveedor LLM '{result['provider']}' devolvió una respuesta vacía."
        )

    return TextEstimationResult(
        estimation=text,
        model=result["model"],
        provider=result["provider"],
        temperature=temperature_for_result(),
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
    version: str = DEFAULT_ESTIMATION_PROMPT_VERSION,
) -> Iterator[str]:
    """Genera una estimación en streaming, cediendo el texto token a token.

    Valida la configuración de forma anticipada (al llamar a la función), de modo
    que los errores de configuración se propagan antes de empezar a consumir el
    generador. Si se pasa `metrics`, se rellena con el modelo, los tokens, el
    coste, si hubo acierto de caché y si se usó el fallback.
    """
    check_description(request.description)
    require_configuration()
    run_input_guardrails(request.description)
    warn_if_temperature_ignored()

    system_prompt, user_message = render_estimation_prompt(request, version=version)
    active_metrics = metrics if metrics is not None else StreamMetrics()
    inner = get_llm_wrapper().complete_stream(
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=settings.temperature,
        max_tokens=settings.llm_max_tokens,
        metrics=active_metrics,
        cache_key=cache_key_material(request, version),
    )
    return _ensure_non_empty(inner)
