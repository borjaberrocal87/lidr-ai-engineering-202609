"""Capa de caché exact-match para respuestas del LLM.

La clave es un SHA-256 del system prompt completo más el mensaje de usuario más
los knobs de generación (modelo, `max_tokens`, temperatura). Cualquier cambio en
el prompt CAG (ejemplos, instrucciones, límite de datos) invalida la caché solo,
sin flush manual.

Se ofrecen tres implementaciones tras la misma interfaz `ResponseCache`:

- `InMemoryCache` (por defecto): TTL en proceso, sin infraestructura.
- `RedisCache` (opcional): persistente y compartida entre procesos.
- `NullCache`: desactiva la caché (`CACHE_BACKEND=none`).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Protocol, cast

import redis
import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

CACHE_KEY_PREFIX = "estimation"


class ResponseCache(Protocol):
    """Interfaz mínima de caché exact-match."""

    def get(self, key: str) -> dict[str, Any] | None: ...

    def set(self, key: str, value: dict[str, Any]) -> None: ...


def make_cache_key(
    *,
    system_prompt: str,
    user_message: str,
    model: str,
    max_tokens: int,
    temperature: float | None,
) -> str:
    """Construye una clave determinista a partir de todo lo que afecta a la salida."""
    payload = json.dumps(
        {
            "system_prompt": system_prompt,
            "user_message": user_message,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{CACHE_KEY_PREFIX}:{digest}"


class NullCache:
    """Caché desactivada: siempre miss, nunca almacena."""

    def get(self, key: str) -> dict[str, Any] | None:
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        return None


class InMemoryCache:
    """Caché TTL en proceso. Cero infraestructura: ideal para desarrollo y tests."""

    def __init__(self, ttl: int = 86_400) -> None:
        self.ttl = ttl
        self._store: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                logger.info("cache.miss")
                return None
            expires_at, payload = entry
            if expires_at < time.time():
                self._store.pop(key, None)
                logger.info("cache.miss", reason="expired")
                return None
        logger.info("cache.hit")
        return payload

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = (time.time() + self.ttl, value)
        logger.info("cache.stored", ttl=self.ttl)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class RedisCache:
    """Caché persistente sobre Redis.

    Falla de forma degradada: si Redis no responde, un `get` se comporta como un
    miss y un `set` no propaga el error, de modo que una caché caída nunca tumba
    la generación de la estimación.
    """

    def __init__(self, client: redis.Redis, ttl: int = 86_400) -> None:
        self.redis = client
        self.ttl = ttl

    @classmethod
    def from_url(cls, url: str, ttl: int = 86_400) -> RedisCache:
        return cls(redis.from_url(url, decode_responses=True), ttl=ttl)

    def get(self, key: str) -> dict[str, Any] | None:
        try:
            cached = self.redis.get(key)
        except redis.RedisError as exc:
            logger.warning("cache.get_failed", error=str(exc))
            return None
        if cached:
            logger.info("cache.hit")
            return cast(dict[str, Any], json.loads(cached))
        logger.info("cache.miss")
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        try:
            self.redis.set(key, json.dumps(value), ex=self.ttl)
            logger.info("cache.stored", ttl=self.ttl)
        except redis.RedisError as exc:
            logger.warning("cache.set_failed", error=str(exc))
