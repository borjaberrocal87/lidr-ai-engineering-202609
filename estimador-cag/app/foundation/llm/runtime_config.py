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


# --- Session 10: runtime overrides for the retrieval toggles -----------------
# Same Redis-hash override pattern as the model knobs, but for the two retrieval
# switches: search mode and reranking. Kept in a SEPARATE hash so resetting the
# model config does not clear them and vice versa.
RETRIEVAL_HASH_KEY = "estimator:runtime_retrieval"

SEARCH_MODE_KEY = "RETRIEVAL_SEARCH_MODE"
RERANK_KEY = "RERANKER_ENABLED"
# Sesión 10 — etapas de la recuperación multifuente. Mismo patrón de override
# que los modelos: el instructor las enciende o apaga desde Ajustes sin reiniciar.
ROUTING_KEY = "RETRIEVAL_ROUTING_ENABLED"
QUERY_PLAN_KEY = "QUERY_PLAN_ENABLED"
FRESHNESS_KEY = "TEMPORAL_DECAY_ENABLED"
# Sesión 10 — knobs numéricos del consenso de horas (no son toggles).
TASK_HOURS_TOP_K_KEY = "TASK_HOURS_TOP_K"
TASK_HOURS_DISTANCE_THRESHOLD_KEY = "TASK_HOURS_DISTANCE_THRESHOLD"
RETRIEVAL_KEYS: tuple[str, ...] = (
    SEARCH_MODE_KEY,
    RERANK_KEY,
    ROUTING_KEY,
    QUERY_PLAN_KEY,
    FRESHNESS_KEY,
    TASK_HOURS_TOP_K_KEY,
    TASK_HOURS_DISTANCE_THRESHOLD_KEY,
)

_VALID_SEARCH_MODES = ("vector", "hybrid")
_TOGGLE_KEYS = (ROUTING_KEY, QUERY_PLAN_KEY, FRESHNESS_KEY)


class RuntimeRetrievalConfig:
    """Overrides respaldados por un hash de Redis para la recuperación S10.

    Las lecturas degradan al default de ``Settings`` si Redis no está: el
    pipeline sigue sirviendo exactamente como lo configuró el ``.env``. Las
    escrituras propagan el fallo para que la API lo mapee a 503, igual que
    ``RuntimeModelConfig``.
    """

    def __init__(self, redis_client: redis.Redis, settings: Settings) -> None:
        self._redis = redis_client
        self._settings = settings

    @classmethod
    def from_url(cls, url: str, settings: Settings) -> RuntimeRetrievalConfig:
        return cls(redis.from_url(url, decode_responses=True), settings)

    def _get_raw(self, key: str) -> str | None:
        try:
            value = self._redis.hget(RETRIEVAL_HASH_KEY, key)
            return str(value) if value is not None else None
        except redis.RedisError as exc:
            log.warning("runtime_retrieval_read_failed", key=key, error=str(exc)[:200])
            return None

    def _set_raw(self, key: str, value: str | None) -> None:
        try:
            if value is None:
                self._redis.hdel(RETRIEVAL_HASH_KEY, key)
            else:
                self._redis.hset(RETRIEVAL_HASH_KEY, key, value)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc

    def effective_search_mode(self) -> str:
        override = self._get_raw(SEARCH_MODE_KEY)
        if override in _VALID_SEARCH_MODES:
            return override
        return self._settings.retrieval_search_mode

    def _effective_bool(self, key: str, default: bool) -> bool:
        override = self._get_raw(key)
        if override is None:
            return default
        return override.lower() == "true"

    def effective_rerank(self) -> bool:
        return self._effective_bool(RERANK_KEY, self._settings.reranker_enabled)

    def effective_routing(self) -> bool:
        return self._effective_bool(ROUTING_KEY, self._settings.retrieval_routing_enabled)

    def effective_query_plan(self) -> bool:
        return self._effective_bool(QUERY_PLAN_KEY, self._settings.retrieval_query_plan_enabled)

    def effective_freshness(self) -> bool:
        return self._effective_bool(FRESHNESS_KEY, self._settings.retrieval_freshness_enabled)

    def set_search_mode(self, value: str | None) -> None:
        if value is not None and value not in _VALID_SEARCH_MODES:
            raise ValueError(f"Invalid search mode: {value}")
        self._set_raw(SEARCH_MODE_KEY, value)

    def set_rerank(self, value: bool | None) -> None:
        self._set_raw(RERANK_KEY, None if value is None else str(value).lower())

    def set_bool(self, key: str, value: bool | None) -> None:
        """Fija uno de los toggles booleanos de la recuperación multifuente."""
        if key not in _TOGGLE_KEYS:
            raise ValueError(f"Unknown retrieval toggle: {key}")
        self._set_raw(key, None if value is None else str(value).lower())

    def effective_task_hours_top_k(self) -> int:
        override = self._get_raw(TASK_HOURS_TOP_K_KEY)
        if override is None:
            return self._settings.task_hours_top_k
        try:
            return int(override)
        except ValueError:
            return self._settings.task_hours_top_k

    def effective_task_hours_distance_threshold(self) -> float:
        override = self._get_raw(TASK_HOURS_DISTANCE_THRESHOLD_KEY)
        if override is None:
            return self._settings.task_hours_distance_threshold
        try:
            return float(override)
        except ValueError:
            return self._settings.task_hours_distance_threshold

    def set_task_hours_top_k(self, value: int | None) -> None:
        if value is not None and value < 1:
            raise ValueError("TASK_HOURS_TOP_K must be >= 1")
        self._set_raw(TASK_HOURS_TOP_K_KEY, None if value is None else str(int(value)))

    def set_task_hours_distance_threshold(self, value: float | None) -> None:
        if value is not None and not (0.0 <= value <= 2.0):
            raise ValueError("TASK_HOURS_DISTANCE_THRESHOLD must be in [0, 2]")
        self._set_raw(
            TASK_HOURS_DISTANCE_THRESHOLD_KEY, None if value is None else str(float(value))
        )

    def snapshot(self) -> dict[str, dict[str, object]]:
        """``{effective, default, overridden}`` por cada knob de retrieval."""
        try:
            overrides = self._redis.hgetall(RETRIEVAL_HASH_KEY)
        except redis.RedisError as exc:
            log.warning("runtime_retrieval_read_failed", key="*", error=str(exc)[:200])
            overrides = {}
        return {
            SEARCH_MODE_KEY: {
                "effective": self.effective_search_mode(),
                "default": self._settings.retrieval_search_mode,
                "overridden": SEARCH_MODE_KEY in overrides,
            },
            RERANK_KEY: {
                "effective": self.effective_rerank(),
                "default": self._settings.reranker_enabled,
                "overridden": RERANK_KEY in overrides,
            },
            ROUTING_KEY: {
                "effective": self.effective_routing(),
                "default": self._settings.retrieval_routing_enabled,
                "overridden": ROUTING_KEY in overrides,
            },
            QUERY_PLAN_KEY: {
                "effective": self.effective_query_plan(),
                "default": self._settings.retrieval_query_plan_enabled,
                "overridden": QUERY_PLAN_KEY in overrides,
            },
            FRESHNESS_KEY: {
                "effective": self.effective_freshness(),
                "default": self._settings.retrieval_freshness_enabled,
                "overridden": FRESHNESS_KEY in overrides,
            },
            TASK_HOURS_TOP_K_KEY: {
                "effective": self.effective_task_hours_top_k(),
                "default": self._settings.task_hours_top_k,
                "overridden": TASK_HOURS_TOP_K_KEY in overrides,
            },
            TASK_HOURS_DISTANCE_THRESHOLD_KEY: {
                "effective": self.effective_task_hours_distance_threshold(),
                "default": self._settings.task_hours_distance_threshold,
                "overridden": TASK_HOURS_DISTANCE_THRESHOLD_KEY in overrides,
            },
        }


__all__ = [
    "FRESHNESS_KEY",
    "HASH_KEY",
    "MODEL_KEYS",
    "MODEL_KEY_TO_SETTING",
    "QUERY_PLAN_KEY",
    "RERANK_KEY",
    "RETRIEVAL_HASH_KEY",
    "RETRIEVAL_KEYS",
    "ROUTING_KEY",
    "SEARCH_MODE_KEY",
    "TASK_HOURS_DISTANCE_THRESHOLD_KEY",
    "TASK_HOURS_TOP_K_KEY",
    "RuntimeConfigUnavailable",
    "RuntimeModelConfig",
    "RuntimeRetrievalConfig",
]
