"""Tests del wrapper de proveedores (LiteLLM Router mockeado, sin red)."""

from types import SimpleNamespace
from typing import Any

import pytest
from openai import APIConnectionError

from app.services.cache import InMemoryCache
from app.services.errors import LLMProviderError
from app.services.llm_wrapper import (
    ROUTER_BALANCED_NAME,
    ROUTER_FALLBACK_NAME,
    ROUTER_PRIMARY_NAME,
    LLMWrapper,
    StreamMetrics,
    _estimate_cost,
    provider_from_model,
)


def _fake_completion(
    model: str,
    content: str = "the answer",
    input_tokens: int = 100,
    output_tokens: int = 50,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


def _fake_stream(*deltas: str) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))])
        for delta in deltas
    ]


def _wrapper(**overrides: Any) -> LLMWrapper:
    kwargs: dict[str, Any] = {
        "primary_model": "gpt-4o-mini",
        "primary_provider": "openai",
        "cache": InMemoryCache(ttl=60),
        "openai_api_key": "fake-openai",
        "anthropic_api_key": "fake-anthropic",
    }
    kwargs.update(overrides)
    return LLMWrapper(**kwargs)


def test_provider_from_model() -> None:
    assert provider_from_model("gpt-4o-mini") == "openai"
    assert provider_from_model("claude-haiku-4-5") == "anthropic"
    assert provider_from_model("openai/llama3.1:8b") == "custom"
    assert provider_from_model("llama3.1:8b") == "custom"


def test_estimate_cost_uses_pricing_table() -> None:
    # 1M input * 0.15 + 1M output * 0.60 = 0.75 USD
    assert _estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert _estimate_cost("desconocido", 1_000_000, 1_000_000) == 0.0


def test_complete_normalises_and_caches(monkeypatch) -> None:
    wrapper = _wrapper()
    calls = {"n": 0}

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        calls["n"] += 1
        return _fake_completion("gpt-4o-mini", content="hello world")

    monkeypatch.setattr(wrapper.router, "completion", fake_completion)

    result = wrapper.complete(
        system_prompt="sys", user_message="usr", temperature=0.2, max_tokens=100
    )

    assert calls["n"] == 1
    assert result["estimation"] == "hello world"
    assert result["model"] == "gpt-4o-mini"
    assert result["provider"] == "openai"
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 50
    assert result["cache_hit"] is False
    assert result["cost_usd"] > 0
    assert result["truncated"] is False

    cached = wrapper.complete(
        system_prompt="sys", user_message="usr", temperature=0.2, max_tokens=100
    )
    assert calls["n"] == 1
    assert cached["cache_hit"] is True
    assert cached["estimation"] == "hello world"


def test_cache_user_message_decouples_key_from_wrapping(monkeypatch) -> None:
    wrapper = _wrapper()
    calls = {"n": 0}

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        calls["n"] += 1
        return _fake_completion("gpt-4o-mini", content="hello")

    monkeypatch.setattr(wrapper.router, "completion", fake_completion)

    first = wrapper.complete(
        system_prompt="sys",
        user_message="<transcripcion-aaaa>hola</transcripcion-aaaa>",
        temperature=0.2,
        max_tokens=100,
        cache_user_message="hola",
    )
    second = wrapper.complete(
        system_prompt="sys",
        user_message="<transcripcion-bbbb>hola</transcripcion-bbbb>",
        temperature=0.2,
        max_tokens=100,
        cache_user_message="hola",
    )

    assert calls["n"] == 1
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True


def test_complete_does_not_cache_truncated(monkeypatch) -> None:
    wrapper = _wrapper()
    calls = {"n": 0}

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        calls["n"] += 1
        return _fake_completion("gpt-4o-mini", finish_reason="length")

    monkeypatch.setattr(wrapper.router, "completion", fake_completion)

    first = wrapper.complete(
        system_prompt="sys", user_message="usr", temperature=0.2, max_tokens=100
    )
    second = wrapper.complete(
        system_prompt="sys", user_message="usr", temperature=0.2, max_tokens=100
    )

    assert first["truncated"] is True
    assert calls["n"] == 2
    assert second["cache_hit"] is False


def test_complete_marks_fallback_used(monkeypatch) -> None:
    wrapper = _wrapper(
        fallback_model="claude-haiku-4-5",
        fallback_provider="anthropic",
    )
    monkeypatch.setattr(
        wrapper.router,
        "completion",
        lambda **kwargs: _fake_completion("claude-haiku-4-5", content="fallback"),
    )

    result = wrapper.complete(
        system_prompt="sys", user_message="usr", temperature=0.2, max_tokens=100
    )

    assert result["fallback_used"] is True
    assert result["provider"] == "anthropic"


def test_fallback_mode_registers_distinct_deployment_names() -> None:
    wrapper = _wrapper(fallback_model="claude-haiku-4-5")

    names = [deployment["model_name"] for deployment in wrapper.router.model_list]

    assert names == [ROUTER_PRIMARY_NAME, ROUTER_FALLBACK_NAME]
    assert wrapper.router_model_name == ROUTER_PRIMARY_NAME


def test_balanced_mode_registers_shared_deployment_name() -> None:
    wrapper = _wrapper(fallback_model="claude-haiku-4-5", routing_mode="balanced")

    names = [deployment["model_name"] for deployment in wrapper.router.model_list]

    assert names == [ROUTER_BALANCED_NAME, ROUTER_BALANCED_NAME]
    assert wrapper.router_model_name == ROUTER_BALANCED_NAME


def test_complete_calls_router_with_mode_specific_model_name(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        seen["model"] = kwargs["model"]
        return _fake_completion("gpt-4o-mini")

    fallback_wrapper = _wrapper(fallback_model="claude-haiku-4-5")
    monkeypatch.setattr(fallback_wrapper.router, "completion", fake_completion)
    fallback_wrapper.complete(
        system_prompt="sys", user_message="usr", temperature=0.2, max_tokens=100
    )
    assert seen["model"] == ROUTER_PRIMARY_NAME

    balanced_wrapper = _wrapper(fallback_model="claude-haiku-4-5", routing_mode="balanced")
    monkeypatch.setattr(balanced_wrapper.router, "completion", fake_completion)
    balanced_wrapper.complete(
        system_prompt="sys", user_message="usr", temperature=0.2, max_tokens=100
    )
    assert seen["model"] == ROUTER_BALANCED_NAME


def test_fallback_used_detects_dated_snapshot(monkeypatch) -> None:
    wrapper = _wrapper(
        fallback_model="claude-haiku-4-5",
        fallback_provider="anthropic",
    )
    monkeypatch.setattr(
        wrapper.router,
        "completion",
        lambda **kwargs: _fake_completion("claude-haiku-4-5-20251001", content="fallback"),
    )

    result = wrapper.complete(
        system_prompt="sys", user_message="usr", temperature=0.2, max_tokens=100
    )

    assert result["fallback_used"] is True
    assert result["provider"] == "anthropic"


def test_cache_hit_preserves_fallback_used(monkeypatch) -> None:
    wrapper = _wrapper(
        fallback_model="claude-haiku-4-5",
        fallback_provider="anthropic",
    )
    monkeypatch.setattr(
        wrapper.router,
        "completion",
        lambda **kwargs: _fake_completion("claude-haiku-4-5-20251001", content="fallback"),
    )

    first = wrapper.complete(
        system_prompt="sys", user_message="usr", temperature=0.2, max_tokens=100
    )
    second = wrapper.complete(
        system_prompt="sys", user_message="usr", temperature=0.2, max_tokens=100
    )

    assert first["fallback_used"] is True
    assert second["cache_hit"] is True
    assert second["fallback_used"] is True


def test_complete_wraps_provider_errors(monkeypatch) -> None:
    wrapper = _wrapper()

    def boom(**kwargs: Any) -> SimpleNamespace:
        raise APIConnectionError(request=None)  # type: ignore[arg-type]

    monkeypatch.setattr(wrapper.router, "completion", boom)

    with pytest.raises(LLMProviderError):
        wrapper.complete(system_prompt="sys", user_message="usr", temperature=0.2, max_tokens=100)


def test_complete_stream_yields_and_caches(monkeypatch) -> None:
    wrapper = _wrapper()
    calls = {"n": 0}

    def fake_completion(**kwargs: Any) -> list[SimpleNamespace]:
        calls["n"] += 1
        return _fake_stream("Hello ", "world", "")

    monkeypatch.setattr(wrapper.router, "completion", fake_completion)

    metrics = StreamMetrics()
    emitted = list(
        wrapper.complete_stream(
            system_prompt="sys",
            user_message="usr",
            temperature=0.2,
            max_tokens=100,
            metrics=metrics,
        )
    )

    assert "".join(emitted) == "Hello world"
    assert metrics.cache_hit is False
    assert metrics.model == "gpt-4o-mini"

    replayed = list(
        wrapper.complete_stream(
            system_prompt="sys",
            user_message="usr",
            temperature=0.2,
            max_tokens=100,
        )
    )
    assert calls["n"] == 1
    assert "".join(replayed) == "Hello world"


def test_anthropic_omits_temperature() -> None:
    wrapper = _wrapper(primary_provider="anthropic", primary_model="claude-haiku-4-5")
    kwargs = wrapper._call_kwargs(
        system_prompt="s", user_message="u", temperature=0.9, max_tokens=10
    )

    assert "temperature" not in kwargs


def test_openai_passes_temperature() -> None:
    wrapper = _wrapper(primary_provider="openai")
    kwargs = wrapper._call_kwargs(
        system_prompt="s", user_message="u", temperature=0.9, max_tokens=10
    )

    assert kwargs["temperature"] == 0.9


def test_custom_provider_uses_openai_prefix_and_base_url() -> None:
    wrapper = _wrapper(
        primary_model="llama3.1:8b",
        primary_provider="custom",
        custom_base_url="http://localhost:11434/v1",
        custom_api_key="custom-key",
    )

    params = wrapper.router.model_list[0]["litellm_params"]
    assert params["model"] == "openai/llama3.1:8b"
    assert params["api_base"] == "http://localhost:11434/v1"
