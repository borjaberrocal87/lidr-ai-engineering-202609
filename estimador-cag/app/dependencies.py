"""Fábricas de dependencias compartidas (caché y wrapper LLM).

Se cachean con `lru_cache` para reutilizar el `Router` de LiteLLM y la conexión
de Redis durante toda la vida del proceso. Los tests pueden sustituir
`get_llm_wrapper` con un doble.
"""

from functools import lru_cache

from app.config import get_settings, reveal_secret
from app.services.cache import InMemoryCache, NullCache, RedisCache, ResponseCache
from app.services.llm_wrapper import LLMWrapper, provider_from_model


@lru_cache
def get_cache() -> ResponseCache:
    settings = get_settings()
    if settings.cache_backend == "none":
        return NullCache()
    if settings.cache_backend == "redis":
        return RedisCache.from_url(settings.redis_url, ttl=settings.cache_ttl)
    return InMemoryCache(ttl=settings.cache_ttl)


@lru_cache
def get_llm_wrapper() -> LLMWrapper:
    settings = get_settings()
    fallback_model = settings.llm_fallback_model
    return LLMWrapper(
        primary_model=settings.llm_model,
        primary_provider=settings.llm_provider,
        fallback_model=fallback_model,
        fallback_provider=provider_from_model(fallback_model) if fallback_model else None,
        routing_mode=settings.llm_routing_mode,
        timeout=settings.llm_timeout_seconds,
        num_retries=settings.llm_max_retries,
        cache=get_cache(),
        openai_api_key=reveal_secret(settings.open_ai_key),
        anthropic_api_key=reveal_secret(settings.anthropic_api_key),
        custom_base_url=settings.custom_llm_base_url,
        custom_api_key=reveal_secret(settings.custom_llm_api_key),
    )
