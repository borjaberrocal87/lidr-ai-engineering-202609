"""Tests for the Session 10 runtime retrieval toggles (fakeredis, no network)."""

from __future__ import annotations

import fakeredis
import redis

from app.config import settings
from app.foundation.llm.runtime_config import (
    RERANK_KEY,
    RETRIEVAL_HASH_KEY,
    SEARCH_MODE_KEY,
    RuntimeRetrievalConfig,
)


def _store() -> RuntimeRetrievalConfig:
    return RuntimeRetrievalConfig(fakeredis.FakeRedis(decode_responses=True), settings)


def test_defaults_come_from_settings() -> None:
    store = _store()

    assert store.effective_search_mode() == settings.retrieval_search_mode
    assert store.effective_rerank() is settings.reranker_enabled


def test_set_and_reset_overrides() -> None:
    store = _store()

    store.set_search_mode("hybrid")
    store.set_rerank(True)

    assert store.effective_search_mode() == "hybrid"
    assert store.effective_rerank() is True

    store.set_search_mode(None)
    store.set_rerank(None)

    assert store.effective_search_mode() == settings.retrieval_search_mode
    assert store.effective_rerank() is settings.reranker_enabled


def test_invalid_search_mode_is_rejected() -> None:
    store = _store()

    try:
        store.set_search_mode("magic")
    except ValueError as exc:
        assert "Invalid search mode" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError")


def test_snapshot_reports_effective_default_and_override() -> None:
    store = _store()
    store.set_search_mode("hybrid")

    snapshot = store.snapshot()

    assert snapshot[SEARCH_MODE_KEY]["effective"] == "hybrid"
    assert snapshot[SEARCH_MODE_KEY]["overridden"] is True
    assert snapshot[RERANK_KEY]["overridden"] is False


def test_snapshot_degrades_when_redis_fails() -> None:
    class BrokenRedis:
        def hgetall(self, key: str) -> dict[str, str]:
            raise redis.RedisError("down")

        def hget(self, key: str, field: str) -> str:
            raise redis.RedisError("down")

    store = RuntimeRetrievalConfig(BrokenRedis(), settings)  # type: ignore[arg-type]

    snapshot = store.snapshot()

    assert snapshot[SEARCH_MODE_KEY]["effective"] == settings.retrieval_search_mode
    assert snapshot[RERANK_KEY]["effective"] is settings.reranker_enabled


def test_hash_key_is_separate_from_model_config() -> None:
    assert RETRIEVAL_HASH_KEY == "estimator:runtime_retrieval"
