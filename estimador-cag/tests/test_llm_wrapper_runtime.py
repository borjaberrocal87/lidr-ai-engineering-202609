"""Tests del wrapper con overrides de modelo en runtime (sin red)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.services.cache import InMemoryCache
from app.services.llm_wrapper import LLMWrapper
from app.services.runtime_config import RuntimeModelConfig


class FakeRuntimeConfig:
    def __init__(self, overrides: dict[str, str] | None = None) -> None:
        self.overrides = overrides or {}

    def effective(self, key: str) -> str:
        defaults = {
            "PRIMARY_MODEL": "gpt-4o-mini",
            "FALLBACK_MODEL": "claude-haiku-4-5",
            "CRITIC_MODEL": "gpt-4o-mini",
            "METADATA_EXTRACTOR_MODEL": "gpt-4o-mini",
            "COMPRESSION_MODEL": "gpt-4o-mini",
            "PROPOSITIONAL_CHUNKER_MODEL": "gpt-4o-mini",
            "CONTEXTUAL_CHUNKER_MODEL": "claude-sonnet-4-5",
        }
        return self.overrides.get(key, defaults[key])

    def is_overridden(self, key: str) -> bool:
        return key in self.overrides


def _wrapper(runtime: RuntimeModelConfig | None) -> LLMWrapper:
    return LLMWrapper(
        primary_model="gpt-4o-mini",
        primary_provider="openai",
        fallback_model="claude-haiku-4-5",
        fallback_provider="anthropic",
        cache=InMemoryCache(ttl=60),
        openai_api_key="fake-openai",
        anthropic_api_key="fake-anthropic",
        runtime_config=runtime,  # type: ignore[arg-type]
    )


def _completion(model: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="hola"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )


def _stream() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ho"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="la"))]),
    ]


def test_primary_model_property_reads_runtime_override() -> None:
    wrapper = _wrapper(FakeRuntimeConfig({"PRIMARY_MODEL": "gpt-4o"}))

    assert wrapper.primary_model == "gpt-4o"
    assert wrapper.fallback_model == "claude-haiku-4-5"


def test_complete_uses_direct_litellm_call_under_override(monkeypatch) -> None:
    wrapper = _wrapper(FakeRuntimeConfig({"PRIMARY_MODEL": "gpt-4o"}))
    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return _completion("gpt-4o")

    monkeypatch.setattr("app.services.llm_wrapper.litellm.completion", fake_completion)

    result = wrapper.complete(
        system_prompt="sys", user_message="usr", temperature=0.2, max_tokens=100
    )

    assert calls[0]["model"] == "gpt-4o"
    assert calls[0]["api_key"] == "fake-openai"
    assert result["model"] == "gpt-4o"
    assert result["estimation"] == "hola"


def test_complete_stream_uses_direct_path_under_override(monkeypatch) -> None:
    wrapper = _wrapper(FakeRuntimeConfig({"PRIMARY_MODEL": "gpt-4o"}))
    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> list[SimpleNamespace]:
        calls.append(kwargs)
        return _stream()

    monkeypatch.setattr("app.services.llm_wrapper.litellm.completion", fake_completion)

    chunks = list(
        wrapper.complete_stream(
            system_prompt="sys", user_message="usr", temperature=0.2, max_tokens=100
        )
    )

    assert calls[0]["stream"] is True
    assert calls[0]["model"] == "gpt-4o"
    assert "".join(chunks) == "hola"
