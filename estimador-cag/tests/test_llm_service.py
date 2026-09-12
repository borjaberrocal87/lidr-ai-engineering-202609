from types import SimpleNamespace

import pytest

from app.config import settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.services.llm_service import (
    EstimationResult,
    LLMConfigurationError,
    build_system_prompt,
    generate_estimation,
)

TRANSCRIPTION = "El cliente necesita una landing page con integración con HubSpot."


def test_build_system_prompt_injects_all_examples() -> None:
    prompt = build_system_prompt()

    for example in ESTIMATION_EXAMPLES:
        assert example["meeting_summary"] in prompt
        assert example["estimation"] in prompt

    assert prompt.count("### Ejemplo") == len(ESTIMATION_EXAMPLES)


def test_unsupported_provider_raises(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "gemini")

    with pytest.raises(LLMConfigurationError):
        generate_estimation(TRANSCRIPTION)


def test_openai_missing_key_raises(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "")

    with pytest.raises(LLMConfigurationError, match="OPEN_AI_KEY"):
        generate_estimation(TRANSCRIPTION)


def test_anthropic_missing_key_raises(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "")

    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        generate_estimation(TRANSCRIPTION)


def test_openai_provider_maps_response(monkeypatch) -> None:
    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content="## Estimación OpenAI")
            usage = SimpleNamespace(prompt_tokens=11, completion_tokens=22)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)], usage=usage
            )

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key=None) -> None:
            captured["api_key"] = api_key
            self.chat = FakeChat()

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    result = generate_estimation(TRANSCRIPTION)

    assert result == EstimationResult(
        estimation="## Estimación OpenAI",
        model="gpt-4o-mini",
        provider="openai",
        input_tokens=11,
        output_tokens=22,
    )
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "gpt-4o-mini"
    assert [m["role"] for m in captured["messages"]] == ["system", "user"]
    assert ESTIMATION_EXAMPLES[0]["estimation"] in captured["messages"][0]["content"]


def test_anthropic_provider_maps_response(monkeypatch) -> None:
    captured: dict = {}

    class FakeMessages:
        def create(self, *, model, max_tokens, system, messages):
            captured.update(
                model=model, max_tokens=max_tokens, system=system, messages=messages
            )
            block = SimpleNamespace(type="text", text="## Estimación Anthropic")
            usage = SimpleNamespace(input_tokens=7, output_tokens=9)
            return SimpleNamespace(content=[block], usage=usage)

    class FakeAnthropic:
        def __init__(self, api_key=None) -> None:
            captured["api_key"] = api_key
            self.messages = FakeMessages()

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "claude-haiku-4-5")
    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)

    result = generate_estimation(TRANSCRIPTION)

    assert result == EstimationResult(
        estimation="## Estimación Anthropic",
        model="claude-haiku-4-5",
        provider="anthropic",
        input_tokens=7,
        output_tokens=9,
    )
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "claude-haiku-4-5"
    assert captured["messages"] == [{"role": "user", "content": TRANSCRIPTION}]
    assert ESTIMATION_EXAMPLES[0]["meeting_summary"] in captured["system"]
