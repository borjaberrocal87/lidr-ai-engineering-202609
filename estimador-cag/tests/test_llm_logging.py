"""Tests del logging estructurado y la trazabilidad del wrapper y la API."""

from types import SimpleNamespace
from typing import Any

import pytest
from openai import APIConnectionError
from starlette.testclient import TestClient
from structlog.testing import capture_logs

from app.services.cache import InMemoryCache
from app.services.errors import LLMProviderError
from app.services.llm_wrapper import LLMWrapper


def _fake_completion(model: str = "gpt-4o-mini", content: str = "## Estimación") -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=22, total_tokens=33),
    )


def _wrapper() -> LLMWrapper:
    return LLMWrapper(
        primary_model="gpt-4o-mini",
        primary_provider="openai",
        cache=InMemoryCache(ttl=60),
        openai_api_key="test-key",
    )


def _events(logs: list[dict]) -> list[str]:
    return [entry["event"] for entry in logs]


def _complete(wrapper: LLMWrapper) -> dict[str, Any]:
    return wrapper.complete(
        system_prompt="sys", user_message="usr", temperature=0.2, max_tokens=100
    )


def test_complete_logs_start_and_end(monkeypatch) -> None:
    wrapper = _wrapper()
    monkeypatch.setattr(wrapper.router, "completion", lambda **kwargs: _fake_completion())

    with capture_logs() as logs:
        _complete(wrapper)

    events = _events(logs)
    assert "llm.call.start" in events
    assert "llm.call.end" in events

    end = next(entry for entry in logs if entry["event"] == "llm.call.end")
    assert end["input_tokens"] == 11
    assert end["output_tokens"] == 22
    assert end["cache_hit"] is False
    assert end["cost_usd"] > 0
    assert isinstance(end["latency_ms"], float)


def test_cache_hit_is_logged(monkeypatch) -> None:
    wrapper = _wrapper()
    monkeypatch.setattr(wrapper.router, "completion", lambda **kwargs: _fake_completion())
    _complete(wrapper)

    with capture_logs() as logs:
        result = _complete(wrapper)

    assert result["cache_hit"] is True
    assert "cache.hit" in _events(logs)
    assert "llm.call.start" not in _events(logs)


def test_complete_error_logs_and_raises(monkeypatch) -> None:
    wrapper = _wrapper()

    def boom(**kwargs: Any) -> SimpleNamespace:
        raise APIConnectionError(request=None)  # type: ignore[arg-type]

    monkeypatch.setattr(wrapper.router, "completion", boom)

    with capture_logs() as logs, pytest.raises(LLMProviderError):
        _complete(wrapper)

    assert "llm.call.error" in _events(logs)


def test_stream_logs_start_end_and_stores(monkeypatch) -> None:
    wrapper = _wrapper()
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="## "))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Estimación"))]),
    ]
    monkeypatch.setattr(wrapper.router, "completion", lambda **kwargs: chunks)

    with capture_logs() as logs:
        list(
            wrapper.complete_stream(
                system_prompt="sys", user_message="usr", temperature=0.2, max_tokens=100
            )
        )

    events = _events(logs)
    assert "llm.stream.start" in events
    assert "llm.stream.end" in events
    assert "cache.stored" in events


def test_logs_do_not_leak_api_key(monkeypatch) -> None:
    secret = "super-secret-api-key"
    wrapper = LLMWrapper(
        primary_model="gpt-4o-mini",
        primary_provider="openai",
        cache=InMemoryCache(ttl=60),
        openai_api_key=secret,
    )
    monkeypatch.setattr(wrapper.router, "completion", lambda **kwargs: _fake_completion())

    with capture_logs() as logs:
        _complete(wrapper)

    assert secret not in str(logs)


def test_middleware_echoes_request_id(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc123"})

    assert response.headers["x-request-id"] == "abc123"


def test_middleware_generates_request_id(client: TestClient) -> None:
    response = client.get("/health")

    assert len(response.headers["x-request-id"]) == 32
