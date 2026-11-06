"""Caché de idempotencia del endpoint de estimación (Sesión 9).

Un cliente puede reintentar ``POST /api/v1/estimate/from-transcript`` con la
misma ``idempotency_key`` y recibir el resultado original sin pagar otra vez el
pipeline LLM. Dos backends intercambiables tras un mismo contrato
(``get`` / ``set`` / ``delete``):

* **Redis** cuando ``REDIS_URL`` está configurado — sobrevive reinicios, se
  comparte entre workers y tiene TTL nativo.
* **Dict en proceso** en caso contrario — fallback con expiración manual para
  tests y desarrollo sin Redis.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import structlog

from app.config import Settings
from app.generation.rag.schemas import GroundedEstimate

log = structlog.get_logger()

_KEY_PREFIX = "idempotency:estimate:"


class IdempotencyStore:
    """Store de estimaciones grounded indexado por idempotency key del cliente."""

    def __init__(self, redis_client: Any | None = None, ttl: int = 86_400) -> None:
        self._redis = redis_client
        self._ttl = ttl
        self._mem: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> IdempotencyStore:
        """Construye el store; prefiere Redis y degrada al dict en memoria."""
        redis_client: Any | None = None
        if settings.redis_url:
            try:
                import redis

                redis_client = redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                )
                redis_client.ping()
            except Exception as exc:
                log.warning(
                    "idempotency.redis_unavailable",
                    reason="setup_failed",
                    error_type=type(exc).__name__,
                )
                redis_client = None
        return cls(redis_client=redis_client, ttl=settings.rag_idempotency_ttl)

    def get(self, key: str) -> GroundedEstimate | None:
        """Devuelve la estimación cacheada para ``key`` o ``None`` si no está."""
        raw = self._get_raw(key)
        if raw is None:
            return None
        try:
            return GroundedEstimate.model_validate_json(raw)
        except Exception:
            self.delete(key)
            return None

    def set(self, key: str, estimate: GroundedEstimate) -> None:
        """Cachea ``estimate`` bajo ``key`` con el TTL configurado."""
        raw = estimate.model_dump_json()
        if self._redis is not None:
            self._redis.set(_KEY_PREFIX + key, raw, ex=self._ttl)
            return
        with self._lock:
            self._mem[key] = (raw, time.monotonic() + self._ttl)

    def delete(self, key: str) -> None:
        """Elimina ``key`` del store (no-op si no existe)."""
        if self._redis is not None:
            self._redis.delete(_KEY_PREFIX + key)
            return
        with self._lock:
            self._mem.pop(key, None)

    def _get_raw(self, key: str) -> str | None:
        if self._redis is not None:
            value = self._redis.get(_KEY_PREFIX + key)
            return str(value) if value is not None else None
        with self._lock:
            entry = self._mem.get(key)
            if entry is None:
                return None
            raw, expires_at = entry
            if time.monotonic() >= expires_at:
                self._mem.pop(key, None)
                return None
            return raw
