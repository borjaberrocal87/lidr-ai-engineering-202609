"""Tests de la capa de caché (memoria, Redis y clave determinista)."""

import fakeredis
import redis

from app.services.cache import (
    InMemoryCache,
    NullCache,
    RedisCache,
    make_cache_key,
)


def _key(**overrides) -> str:
    args = {
        "system_prompt": "sys",
        "user_message": "usr",
        "model": "gpt-4o-mini",
        "max_tokens": 100,
        "temperature": 0.2,
    }
    args.update(overrides)
    return make_cache_key(**args)


def test_make_cache_key_is_deterministic() -> None:
    assert _key() == _key()


def test_make_cache_key_changes_when_inputs_change() -> None:
    base = _key()
    assert _key(system_prompt="otro") != base
    assert _key(user_message="otro") != base
    assert _key(model="gpt-4o") != base
    assert _key(max_tokens=200) != base
    assert _key(temperature=0.9) != base


def test_in_memory_set_then_get_roundtrips() -> None:
    cache = InMemoryCache(ttl=60)
    payload = {"estimation": "...", "model": "gpt-4o-mini"}

    cache.set(_key(), payload)

    assert cache.get(_key()) == payload


def test_in_memory_get_returns_none_on_miss() -> None:
    assert InMemoryCache(ttl=60).get(_key()) is None


def test_in_memory_expires_entry() -> None:
    cache = InMemoryCache(ttl=-1)
    cache.set(_key(), {"estimation": "vieja"})

    assert cache.get(_key()) is None


def test_null_cache_never_stores() -> None:
    cache = NullCache()
    cache.set(_key(), {"estimation": "x"})

    assert cache.get(_key()) is None


def test_redis_set_then_get_roundtrips() -> None:
    cache = RedisCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)
    payload = {"estimation": "...", "cost_usd": 0.001}

    cache.set(_key(), payload)

    assert cache.get(_key()) == payload


def test_redis_get_returns_none_on_miss() -> None:
    cache = RedisCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)

    assert cache.get(_key()) is None


def test_redis_applies_ttl() -> None:
    cache = RedisCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)
    cache.set(_key(), {"x": 1})

    ttl = cache.redis.ttl(_key())
    assert 0 < ttl <= 60


def test_redis_get_failure_degrades_to_miss() -> None:
    cache = RedisCache(_BrokenRedis(), ttl=60)

    assert cache.get(_key()) is None


def test_redis_set_failure_does_not_raise() -> None:
    cache = RedisCache(_BrokenRedis(), ttl=60)

    cache.set(_key(), {"x": 1})


class _BrokenRedis:
    """Cliente Redis que siempre falla, para verificar la degradación."""

    def get(self, key: str) -> None:
        raise redis.RedisError("redis caído")

    def set(self, key: str, value: str, ex: int) -> None:
        raise redis.RedisError("redis caído")
