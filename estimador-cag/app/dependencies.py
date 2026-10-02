"""Fábricas de dependencias compartidas (caché y wrapper LLM).

Se cachean con `lru_cache` para reutilizar el `Router` de LiteLLM y la conexión
de Redis durante toda la vida del proceso. Los tests pueden sustituir
`get_llm_wrapper` con un doble.
"""

from functools import lru_cache

import redis
import structlog
from openai import OpenAI

from app.cache.semantic import EstimationSemanticCache
from app.config import get_settings, reveal_secret
from app.services.cache import InMemoryCache, NullCache, RedisCache, ResponseCache
from app.services.llm_wrapper import LLMWrapper, provider_from_model

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


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


@lru_cache
def get_openai_client() -> OpenAI | None:
    """Cliente OpenAI perezoso para la moderación y los embeddings.

    Devuelve ``None`` si no hay clave o si la moderación está desactivada, de
    modo que el pipeline sigue funcionando solo con las capas regex.
    """
    settings = get_settings()
    if not settings.has_moderation:
        return None
    key = reveal_secret(settings.open_ai_key)
    if not key:
        return None
    return OpenAI(api_key=key)


@lru_cache
def get_semantic_cache() -> EstimationSemanticCache | None:
    """Construye la caché semántica, degradando si falta algo.

    Devuelve ``None`` (con warning) si está desactivada, si no hay embeddings
    configurados o si Redis Stack / RediSearch no está disponible. Así la
    generación sigue funcionando sin la capa semántica.
    """
    settings = get_settings()
    if not settings.semantic_cache_enabled or not settings.has_embeddings:
        return None

    try:
        from redisvl.utils.vectorize import OpenAITextVectorizer

        api_key = reveal_secret(settings.embedding_api_key) or reveal_secret(settings.open_ai_key)
        api_config: dict[str, str] = {"api_key": api_key or ""}
        if settings.embedding_base_url:
            api_config["base_url"] = settings.embedding_base_url
        vectorizer = OpenAITextVectorizer(model=settings.embedding_model, api_config=api_config)

        redis_client = redis.from_url(settings.redis_url, decode_responses=False)
        return EstimationSemanticCache(
            redis_client=redis_client,
            vectorizer=vectorizer,
            threshold=settings.semantic_cache_threshold,
            ttl=settings.semantic_cache_ttl,
            log_only=settings.semantic_cache_log_only,
        )
    except Exception as exc:
        logger.warning(
            "semantic_cache_disabled",
            reason="setup_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return None
