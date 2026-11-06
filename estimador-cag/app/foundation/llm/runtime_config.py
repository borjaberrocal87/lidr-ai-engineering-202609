"""Configuración de modelos LLM mutable en runtime, respaldada por Redis.

``Settings`` (.env) sigue siendo la capa inmutable de *defaults*; este store
guarda *overrides* como un hash de Redis, de modo que se puede cambiar de modelo
desde Ajustes en mitad de una sesión sin editar ``.env`` ni recrear contenedores.
Los overrides sobreviven a reinicios (Redis es la fuente de verdad y se lee en
cada llamada).

Semántica de fallos:
- Lecturas degradadas: si Redis no está, ``effective()`` devuelve el default de
  settings y el servicio se comporta como si no hubiera overrides.
- Escrituras: propagan ``RuntimeConfigUnavailable`` para que la API lo mapee a
  un 503 y el fallo sea visible.
"""

from __future__ import annotations

import redis
import structlog

from app.config import Settings

log = structlog.get_logger()

# Claves públicas del API. Mapean a campos de ``Settings`` en minúsculas.
MODEL_KEY_TO_SETTING: dict[str, str] = {
    "PRIMARY_MODEL": "llm_model",
    "FALLBACK_MODEL": "llm_fallback_model",
    "CRITIC_MODEL": "critic_model",
    "METADATA_EXTRACTOR_MODEL": "metadata_extractor_model",
    "COMPRESSION_MODEL": "compression_model",
    "PROPOSITIONAL_CHUNKER_MODEL": "propositional_chunker_model",
    "CONTEXTUAL_CHUNKER_MODEL": "contextual_chunker_model",
}

MODEL_KEYS: tuple[str, ...] = tuple(MODEL_KEY_TO_SETTING)

# Claves que heredan del modelo primario cuando el campo está vacío.
PRIMARY_INHERIT_KEYS = {
    "CRITIC_MODEL",
    "METADATA_EXTRACTOR_MODEL",
    "COMPRESSION_MODEL",
}

HASH_KEY = "estimator:runtime_config"


class RuntimeConfigUnavailable(Exception):
    """Se lanza cuando falla una ESCRITURA de override (Redis inaccesible)."""


class RuntimeModelConfig:
    """Store de overrides de modelo respaldado por un hash de Redis."""

    def __init__(self, redis_client: redis.Redis, settings: Settings) -> None:
        self._redis = redis_client
        self._settings = settings

    @classmethod
    def from_url(cls, url: str, settings: Settings) -> RuntimeModelConfig:
        return cls(redis.from_url(url, decode_responses=True), settings)

    def get(self, key: str) -> str | None:
        """Devuelve el override de ``key`` o ``None`` (sin override / Redis caído)."""
        self._validate_key(key)
        try:
            value = self._redis.hget(HASH_KEY, key)
            return str(value) if value is not None else None
        except redis.RedisError as exc:
            log.warning("runtime_config_read_failed", key=key, error=str(exc)[:200])
            return None

    def set(self, key: str, value: str | None) -> None:
        """Fija el override; ``None`` lo borra (vuelve al default de .env)."""
        self._validate_key(key)
        try:
            if value is None:
                self._redis.hdel(HASH_KEY, key)
            else:
                self._redis.hset(HASH_KEY, key, value)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc

    def effective(self, key: str) -> str:
        """Valor que debe usar el pipeline: override si existe, si no el default."""
        return self.get(key) or self.default(key)

    def default(self, key: str) -> str:
        """Default de settings, resolviendo la herencia del primario."""
        self._validate_key(key)
        raw = getattr(self._settings, MODEL_KEY_TO_SETTING[key])
        if key in PRIMARY_INHERIT_KEYS and not raw:
            return self._settings.llm_model
        return str(raw)

    def is_overridden(self, key: str) -> bool:
        return self.get(key) is not None

    def snapshot(self) -> dict[str, dict[str, str | bool]]:
        """Una entrada por knob: ``{effective, default, overridden}``."""
        try:
            overrides = self._redis.hgetall(HASH_KEY)
        except redis.RedisError as exc:
            log.warning("runtime_config_read_failed", key="*", error=str(exc)[:200])
            overrides = {}
        return {
            key: {
                "effective": str(overrides.get(key) or self.default(key)),
                "default": self.default(key),
                "overridden": key in overrides,
            }
            for key in MODEL_KEYS
        }

    def reset_all(self) -> None:
        """Borra todos los overrides (vuelve a los defaults de .env)."""
        try:
            self._redis.delete(HASH_KEY)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc

    @staticmethod
    def _validate_key(key: str) -> None:
        if key not in MODEL_KEY_TO_SETTING:
            raise ValueError(f"Clave de modelo desconocida: {key}")


__all__ = [
    "HASH_KEY",
    "MODEL_KEYS",
    "MODEL_KEY_TO_SETTING",
    "RuntimeConfigUnavailable",
    "RuntimeModelConfig",
]
