"""Tests for the Session 10 runtime retrieval toggles (fakeredis, no network)."""

from __future__ import annotations

import fakeredis
import redis

from app.config import settings
from app.foundation.llm.runtime_config import (
    FRESHNESS_KEY,
    QUERY_PLAN_KEY,
    RERANK_KEY,
    RETRIEVAL_HASH_KEY,
    ROUTING_KEY,
    SEARCH_MODE_KEY,
    TASK_HOURS_DISTANCE_THRESHOLD_KEY,
    TASK_HOURS_TOP_K_KEY,
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


def test_multisource_defaults_come_from_settings() -> None:
    store = _store()

    assert store.effective_routing() is settings.retrieval_routing_enabled
    assert store.effective_query_plan() is settings.retrieval_query_plan_enabled
    assert store.effective_freshness() is settings.retrieval_freshness_enabled
    assert store.effective_task_hours_top_k() == settings.task_hours_top_k
    assert store.effective_task_hours_distance_threshold() == settings.task_hours_distance_threshold


def test_multisource_toggles_can_be_set_and_reset() -> None:
    store = _store()

    store.set_bool(ROUTING_KEY, False)
    store.set_bool(QUERY_PLAN_KEY, False)
    store.set_bool(FRESHNESS_KEY, True)

    assert store.effective_routing() is False
    assert store.effective_query_plan() is False
    assert store.effective_freshness() is True

    store.set_bool(ROUTING_KEY, None)
    store.set_bool(QUERY_PLAN_KEY, None)
    store.set_bool(FRESHNESS_KEY, None)

    assert store.effective_routing() is settings.retrieval_routing_enabled
    assert store.effective_query_plan() is settings.retrieval_query_plan_enabled
    assert store.effective_freshness() is settings.retrieval_freshness_enabled


def test_task_hours_knobs_are_validated() -> None:
    store = _store()

    store.set_task_hours_top_k(8)
    store.set_task_hours_distance_threshold(0.3)

    assert store.effective_task_hours_top_k() == 8
    assert store.effective_task_hours_distance_threshold() == 0.3

    try:
        store.set_task_hours_top_k(0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError")

    try:
        store.set_task_hours_distance_threshold(9.0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError")


def test_snapshot_includes_multisource_knobs() -> None:
    snapshot = _store().snapshot()

    assert ROUTING_KEY in snapshot
    assert QUERY_PLAN_KEY in snapshot
    assert FRESHNESS_KEY in snapshot
    assert TASK_HOURS_TOP_K_KEY in snapshot
    assert TASK_HOURS_DISTANCE_THRESHOLD_KEY in snapshot
