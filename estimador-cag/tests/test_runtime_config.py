"""Tests del store de overrides de modelo en runtime (fakeredis, sin red)."""

from __future__ import annotations

import fakeredis
import redis

from app.config import settings
from app.services.runtime_config import (
    HASH_KEY,
    MODEL_KEYS,
    RuntimeModelConfig,
)


def _store() -> RuntimeModelConfig:
    return RuntimeModelConfig(fakeredis.FakeRedis(decode_responses=True), settings)


def test_snapshot_has_one_entry_per_knob() -> None:
    snapshot = _store().snapshot()

    assert set(snapshot) == set(MODEL_KEYS)
    assert snapshot["PRIMARY_MODEL"]["default"] == settings.llm_model
    assert snapshot["PRIMARY_MODEL"]["overridden"] is False


def test_set_and_reset_override() -> None:
    store = _store()

    store.set("PRIMARY_MODEL", "gpt-4o")

    assert store.get("PRIMARY_MODEL") == "gpt-4o"
    assert store.effective("PRIMARY_MODEL") == "gpt-4o"
    assert store.is_overridden("PRIMARY_MODEL") is True

    store.set("PRIMARY_MODEL", None)

    assert store.get("PRIMARY_MODEL") is None
    assert store.effective("PRIMARY_MODEL") == settings.llm_model
    assert store.is_overridden("PRIMARY_MODEL") is False


def test_inherited_knobs_fall_back_to_primary() -> None:
    store = _store()

    assert store.default("CRITIC_MODEL") == settings.llm_model
    assert store.effective("METADATA_EXTRACTOR_MODEL") == settings.llm_model

    store.set("CRITIC_MODEL", "gpt-4o")

    assert store.effective("CRITIC_MODEL") == "gpt-4o"
    assert store.default("CRITIC_MODEL") == settings.llm_model


def test_reset_all_drops_overrides() -> None:
    store = _store()
    store.set("PRIMARY_MODEL", "gpt-4o")
    store.set("CRITIC_MODEL", "gpt-4o")

    store.reset_all()

    assert store.snapshot()["PRIMARY_MODEL"]["overridden"] is False
    assert store.snapshot()["CRITIC_MODEL"]["overridden"] is False


def test_unknown_key_is_rejected() -> None:
    store = _store()

    try:
        store.set("NOT_A_MODEL", "gpt-4o")
    except ValueError as exc:
        assert "desconocida" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Se esperaba ValueError")


def test_snapshot_degrades_when_redis_fails() -> None:
    class BrokenRedis:
        def hgetall(self, key: str) -> dict[str, str]:
            raise redis.RedisError("down")

    store = RuntimeModelConfig(BrokenRedis(), settings)  # type: ignore[arg-type]
    snapshot = store.snapshot()

    assert snapshot["PRIMARY_MODEL"]["effective"] == settings.llm_model
    assert snapshot["PRIMARY_MODEL"]["overridden"] is False


def test_hash_key_is_stable() -> None:
    assert HASH_KEY == "estimator:runtime_config"
