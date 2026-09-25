"""Wrapper de abstracción de proveedores sobre LiteLLM.

Concentra en un único punto todo lo transversal a la llamada al LLM:

- **Selección de proveedor** mediante un `Router`. En modo `fallback` usa un
  modelo primario y salta al secundario solo si el primario falla; en modo
  `balanced` reparte las peticiones entre ambos. Ver `LLM_ROUTING_MODE`.
- **Caché exact-match** de la respuesta (`ResponseCache`), con flag `cache_hit`.
- **Coste estimado** a partir de una tabla de precios por millón de tokens.
- **Métricas** de la generación (tokens, latencia, truncamiento, fallback).

La lógica de negocio (construcción del prompt CAG, validación de la entrada,
mapeo al contrato del servicio) permanece en `llm_service.py`; el wrapper solo
habla con el proveedor.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import structlog
from litellm.router import Router
from openai import OpenAIError

from app.services.cache import ResponseCache, make_cache_key
from app.services.errors import LLMProviderError

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

Provider = Literal["openai", "anthropic", "custom"]
RoutingMode = Literal["fallback", "balanced"]

# En modo `fallback` cada deployment tiene su propio `model_name` y el Router
# salta del primario al secundario solo si el primario falla. En modo `balanced`
# ambos comparten el mismo `model_name` y LiteLLM reparte las peticiones
# (routing_strategy por defecto: simple-shuffle).
ROUTER_PRIMARY_NAME = "primary"
ROUTER_FALLBACK_NAME = "fallback"
ROUTER_BALANCED_NAME = "estimator"

_DATE_SUFFIX = re.compile(r"-\d{8}$")

# Coste por 1M de tokens (USD). Actualízalo cuando cambien los precios.
MODEL_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "deepseek-v4-flash": {"input": 0.30, "output": 1.12},
}


@dataclass
class StreamMetrics:
    """Métricas de una generación en streaming, rellenadas al agotar el generador."""

    model: str = ""
    provider: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    truncated: bool = False
    cost_usd: float | None = None
    cache_hit: bool = False
    fallback_used: bool = False


def _normalise_model_name(model: str) -> str:
    """Elimina prefijos de proveedor (``openai/``) que LiteLLM puede devolver."""
    return model.split("/", 1)[1] if "/" in model else model


def _canonical_model_name(model: str) -> str:
    """Nombre comparable: sin prefijo de proveedor ni sufijo de fecha.

    Anthropic devuelve el snapshot fechado (``claude-haiku-4-5-20251001``)
    aunque el deployment se registre con el alias (``claude-haiku-4-5``).
    """
    return _DATE_SUFFIX.sub("", _normalise_model_name(model)).lower()


def provider_from_model(model: str) -> Provider:
    """Infiere el proveedor a partir del nombre del modelo."""
    name = _normalise_model_name(model).lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    return "custom"


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    base = _normalise_model_name(model)
    costs = MODEL_COSTS.get(base) or MODEL_COSTS.get(model) or {"input": 0.0, "output": 0.0}
    return round((tokens_in * costs["input"] + tokens_out * costs["output"]) / 1_000_000, 6)


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)


def _extract_delta(chunk: Any) -> str:
    """Extrae el delta de texto de un chunk de streaming de LiteLLM."""
    try:
        delta = chunk.choices[0].delta
    except (AttributeError, IndexError):
        return ""
    content = getattr(delta, "content", None)
    return content or ""


class LLMWrapper:
    """Cliente LLM unificado con fallback, caché, coste y logging."""

    def __init__(
        self,
        *,
        primary_model: str,
        cache: ResponseCache,
        primary_provider: Provider = "openai",
        fallback_model: str = "",
        fallback_provider: Provider | None = None,
        routing_mode: RoutingMode = "fallback",
        timeout: float = 30.0,
        num_retries: int = 2,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        custom_base_url: str = "",
        custom_api_key: str | None = None,
    ) -> None:
        self.primary_model = primary_model
        self.primary_provider = primary_provider
        self.fallback_model = fallback_model
        self.routing_mode = routing_mode
        self.cache = cache

        if routing_mode == "balanced":
            primary_name = fallback_name = ROUTER_BALANCED_NAME
        else:
            primary_name, fallback_name = ROUTER_PRIMARY_NAME, ROUTER_FALLBACK_NAME
        self.router_model_name = primary_name

        deployments = [
            self._deployment(
                model_name=primary_name,
                model=primary_model,
                provider=primary_provider,
                timeout=timeout,
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                custom_base_url=custom_base_url,
                custom_api_key=custom_api_key,
            )
        ]
        fallbacks: list[dict[str, list[str]]] = []
        if fallback_model:
            active_fallback_provider = fallback_provider or provider_from_model(fallback_model)
            deployments.append(
                self._deployment(
                    model_name=fallback_name,
                    model=fallback_model,
                    provider=active_fallback_provider,
                    timeout=timeout,
                    openai_api_key=openai_api_key,
                    anthropic_api_key=anthropic_api_key,
                    custom_base_url=custom_base_url,
                    custom_api_key=custom_api_key,
                )
            )
            fallbacks = [{primary_name: [fallback_name]}]

        self.router = Router(
            model_list=deployments,
            fallbacks=fallbacks,
            num_retries=num_retries,
        )

    @staticmethod
    def _deployment(
        *,
        model_name: str,
        model: str,
        provider: Provider,
        timeout: float,
        openai_api_key: str | None,
        anthropic_api_key: str | None,
        custom_base_url: str,
        custom_api_key: str | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"model": model, "timeout": timeout}
        if provider == "openai":
            params["api_key"] = openai_api_key
        elif provider == "anthropic":
            params["api_key"] = anthropic_api_key
        else:
            # Servidores OpenAI-compatible (Ollama, vLLM, LM Studio, NaN API...).
            params["model"] = f"openai/{model}"
            params["api_key"] = custom_api_key
            params["api_base"] = custom_base_url
        return {"model_name": model_name, "litellm_params": params}

    def _cache_key(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None,
        max_tokens: int,
        cache_user_message: str | None = None,
    ) -> str:
        """Clave de caché sobre el contenido lógico, no sobre el artefacto enviado.

        `cache_user_message` permite hashear el mensaje canónico (p. ej. la
        transcripción sin el nonce aleatorio de `_wrap_transcription`), aunque al
        modelo se le envíe `user_message`. Sin él, el nonce cambiaría en cada
        petición y la caché nunca acertaría.
        """
        keyed_message = user_message if cache_user_message is None else cache_user_message
        return make_cache_key(
            system_prompt=system_prompt,
            user_message=keyed_message,
            model=self.primary_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _call_kwargs(
        self,
        *,
        system_prompt: str,
        user_message: str,
        temperature: float | None,
        max_tokens: int,
        stream: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.router_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
        }
        if stream:
            kwargs["stream"] = True
        # Anthropic ignora `temperature` en este proyecto (su SDK no lo acepta).
        if temperature is not None and self.primary_provider != "anthropic":
            kwargs["temperature"] = temperature
        return kwargs

    def complete(
        self,
        *,
        system_prompt: str,
        user_message: str,
        temperature: float | None,
        max_tokens: int,
        cache_user_message: str | None = None,
    ) -> dict[str, Any]:
        """Llamada bloqueante con caché y fallback. Devuelve la respuesta normalizada."""
        key = self._cache_key(
            system_prompt, user_message, temperature, max_tokens, cache_user_message
        )
        cached = self.cache.get(key)
        if cached is not None:
            # Se preservan `fallback_used` y `cost_usd` originales: un hit de caché
            # no vuelve a llamar al proveedor, pero sí recuerda quién lo atendió.
            return {**cached, "cache_hit": True}

        log = logger.bind(call_id=uuid.uuid4().hex[:12])
        log.info(
            "llm.call.start",
            model=self.primary_model,
            provider=self.primary_provider,
            max_tokens=max_tokens,
        )
        started_at = time.perf_counter()
        kwargs = self._call_kwargs(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            response = self.router.completion(**kwargs)
        except OpenAIError as exc:
            log.error(
                "llm.call.error",
                error_type=type(exc).__name__,
                latency_ms=_elapsed_ms(started_at),
            )
            raise LLMProviderError(f"Fallo del proveedor LLM '{self.primary_provider}'.") from exc

        result = self._normalise(response, latency_ms=_elapsed_ms(started_at))
        log.info(
            "llm.call.end",
            model=result["model"],
            provider=result["provider"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            cost_usd=result["cost_usd"],
            fallback_used=result["fallback_used"],
            latency_ms=result["latency_ms"],
            cache_hit=False,
        )
        if not result["truncated"] and result["estimation"]:
            self.cache.set(key, result)
        return {**result, "cache_hit": False}

    def complete_stream(
        self,
        *,
        system_prompt: str,
        user_message: str,
        temperature: float | None,
        max_tokens: int,
        metrics: StreamMetrics | None = None,
        cache_user_message: str | None = None,
    ) -> Iterator[str]:
        """Generador de texto con caché y fallback.

        En un acierto de caché, reproduce la estimación completa como un único
        chunk para que la UX del cliente no cambie.
        """
        key = self._cache_key(
            system_prompt, user_message, temperature, max_tokens, cache_user_message
        )
        cached = self.cache.get(key)
        if cached is not None:
            text = str(cached.get("estimation", ""))
            if metrics is not None:
                metrics.model = str(cached.get("model", self.primary_model))
                metrics.provider = str(cached.get("provider", self.primary_provider))
                metrics.input_tokens = cached.get("input_tokens")
                metrics.output_tokens = cached.get("output_tokens")
                metrics.truncated = bool(cached.get("truncated", False))
                metrics.cost_usd = cached.get("cost_usd", 0.0)
                metrics.cache_hit = True
                metrics.fallback_used = bool(cached.get("fallback_used", False))
            logger.info("llm.stream.cache_hit", chars=len(text))
            yield text
            return

        log = logger.bind(call_id=uuid.uuid4().hex[:12])
        log.info(
            "llm.stream.start",
            model=self.primary_model,
            provider=self.primary_provider,
            max_tokens=max_tokens,
        )
        started_at = time.perf_counter()
        kwargs = self._call_kwargs(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        parts: list[str] = []
        stream_model = self.primary_model
        stream_usage: Any = None
        try:
            response = self.router.completion(**kwargs)
            for chunk in response:
                chunk_model = getattr(chunk, "model", None)
                if chunk_model:
                    stream_model = _normalise_model_name(str(chunk_model))
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    stream_usage = chunk_usage
                delta = _extract_delta(chunk)
                if delta:
                    parts.append(delta)
                    yield delta
        except OpenAIError as exc:
            log.error(
                "llm.stream.error",
                error_type=type(exc).__name__,
                latency_ms=_elapsed_ms(started_at),
            )
            raise LLMProviderError(f"Fallo del proveedor LLM '{self.primary_provider}'.") from exc

        rendered = "".join(parts)
        latency_ms = _elapsed_ms(started_at)
        input_tokens = getattr(stream_usage, "prompt_tokens", None) if stream_usage else None
        output_tokens = getattr(stream_usage, "completion_tokens", None) if stream_usage else None
        provider = provider_from_model(stream_model)
        fallback_used = (
            bool(self.fallback_model)
            and _canonical_model_name(stream_model) == _canonical_model_name(self.fallback_model)
            and _canonical_model_name(stream_model) != _canonical_model_name(self.primary_model)
        )
        cost_usd = _estimate_cost(stream_model, input_tokens or 0, output_tokens or 0)
        if metrics is not None:
            metrics.model = stream_model
            metrics.provider = provider
            metrics.input_tokens = input_tokens
            metrics.output_tokens = output_tokens
            metrics.truncated = False
            metrics.cost_usd = cost_usd
            metrics.cache_hit = False
            metrics.fallback_used = fallback_used
        log.info(
            "llm.stream.end",
            chars=len(rendered),
            model=stream_model,
            fallback_used=fallback_used,
            latency_ms=latency_ms,
        )

        if rendered:
            self.cache.set(
                key,
                {
                    "estimation": rendered,
                    "model": stream_model,
                    "provider": provider,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "truncated": False,
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "fallback_used": fallback_used,
                },
            )

    def _normalise(self, response: Any, *, latency_ms: float) -> dict[str, Any]:
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        text = (getattr(message, "content", None) or "") if message is not None else ""
        finish_reason = (getattr(choice, "finish_reason", None) or "stop").lower()

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)

        raw_model = getattr(response, "model", None) or self.primary_model
        model = _normalise_model_name(str(raw_model))
        fallback_used = (
            bool(self.fallback_model)
            and _canonical_model_name(model) == _canonical_model_name(self.fallback_model)
            and _canonical_model_name(model) != _canonical_model_name(self.primary_model)
        )

        return {
            "estimation": text,
            "model": model,
            "provider": provider_from_model(model),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "truncated": finish_reason == "length",
            "latency_ms": latency_ms,
            "cost_usd": _estimate_cost(model, input_tokens or 0, output_tokens or 0),
            "fallback_used": fallback_used,
        }
