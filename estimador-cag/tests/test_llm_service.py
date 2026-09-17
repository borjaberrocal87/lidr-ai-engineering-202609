from types import SimpleNamespace

import pytest

from app.config import settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.services.llm_service import (
    EstimationResult,
    LLMConfigurationError,
    StreamMetrics,
    build_system_prompt,
    generate_estimation,
    stream_estimation,
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

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            usage = SimpleNamespace(input_tokens=11, output_tokens=22)
            return SimpleNamespace(output_text="## Estimación OpenAI", usage=usage)

    class FakeOpenAI:
        def __init__(self, api_key=None) -> None:
            captured["api_key"] = api_key
            self.responses = FakeResponses()

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "temperature", 0.3)
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    result = generate_estimation(TRANSCRIPTION)

    assert result == EstimationResult(
        estimation="## Estimación OpenAI",
        model="gpt-4o-mini",
        provider="openai",
        temperature=0.3,
        input_tokens=11,
        output_tokens=22,
    )
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["temperature"] == 0.3
    assert captured["input"] == TRANSCRIPTION
    assert ESTIMATION_EXAMPLES[0]["estimation"] in captured["instructions"]


def test_anthropic_provider_maps_response(monkeypatch) -> None:
    captured: dict = {}

    class FakeMessages:
        def create(self, *, model, max_tokens, system, messages):
            captured.update(model=model, max_tokens=max_tokens, system=system, messages=messages)
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


def test_stream_missing_key_raises_eagerly(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "")

    with pytest.raises(LLMConfigurationError, match="OPEN_AI_KEY"):
        stream_estimation(TRANSCRIPTION)


def test_stream_unsupported_provider_raises(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "gemini")

    with pytest.raises(LLMConfigurationError):
        stream_estimation(TRANSCRIPTION)


def test_openai_stream_yields_deltas_and_metrics(monkeypatch) -> None:
    captured: dict = {}

    class FakeOpenAIStream:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> bool:
            return False

        def __iter__(self):
            yield SimpleNamespace(type="response.output_text.delta", delta="## ")
            yield SimpleNamespace(type="response.output_text.delta", delta="Estimación")
            yield SimpleNamespace(type="response.completed")

        def get_final_response(self):
            usage = SimpleNamespace(input_tokens=5, output_tokens=7)
            return SimpleNamespace(usage=usage)

    class FakeResponses:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return FakeOpenAIStream()

    class FakeOpenAI:
        def __init__(self, api_key=None) -> None:
            captured["api_key"] = api_key
            self.responses = FakeResponses()

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "temperature", 0.2)
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    metrics = StreamMetrics(model="", provider="")
    chunks = list(stream_estimation(TRANSCRIPTION, metrics))

    assert chunks == ["## ", "Estimación"]
    assert "".join(chunks) == "## Estimación"
    assert metrics == StreamMetrics(
        model="gpt-4o-mini",
        provider="openai",
        input_tokens=5,
        output_tokens=7,
    )
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["input"] == TRANSCRIPTION
    assert ESTIMATION_EXAMPLES[0]["estimation"] in captured["instructions"]


def test_anthropic_stream_yields_deltas_and_metrics(monkeypatch) -> None:
    captured: dict = {}

    class FakeAnthropicStream:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> bool:
            return False

        def __stream_text__(self):
            yield "## "
            yield "Estimación"

        def get_final_message(self):
            usage = SimpleNamespace(input_tokens=3, output_tokens=4)
            return SimpleNamespace(usage=usage)

    class FakeMessages:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return FakeAnthropicStream()

    class FakeAnthropic:
        def __init__(self, api_key=None) -> None:
            captured["api_key"] = api_key
            self.messages = FakeMessages()

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "claude-haiku-4-5")
    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)

    metrics = StreamMetrics(model="", provider="")
    chunks = list(stream_estimation(TRANSCRIPTION, metrics))

    assert chunks == ["## ", "Estimación"]
    assert "".join(chunks) == "## Estimación"
    assert metrics == StreamMetrics(
        model="claude-haiku-4-5",
        provider="anthropic",
        input_tokens=3,
        output_tokens=4,
    )
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "claude-haiku-4-5"
    assert captured["messages"] == [{"role": "user", "content": TRANSCRIPTION}]
    assert ESTIMATION_EXAMPLES[0]["meeting_summary"] in captured["system"]


def test_custom_missing_base_url_raises(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "custom")
    monkeypatch.setattr(settings, "custom_llm_base_url", "")
    monkeypatch.setattr(settings, "custom_llm_api_key", "test-key")

    with pytest.raises(LLMConfigurationError, match="CUSTOM_LLM_BASE_URL"):
        generate_estimation(TRANSCRIPTION)


def test_custom_missing_key_raises(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "custom")
    monkeypatch.setattr(settings, "custom_llm_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(settings, "custom_llm_api_key", "")

    with pytest.raises(LLMConfigurationError, match="CUSTOM_LLM_API_KEY"):
        generate_estimation(TRANSCRIPTION)


def test_custom_provider_maps_response(monkeypatch) -> None:
    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content="## Estimación Custom")
            usage = SimpleNamespace(prompt_tokens=13, completion_tokens=21)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = FakeChat()

    monkeypatch.setattr(settings, "llm_provider", "custom")
    monkeypatch.setattr(settings, "custom_llm_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(settings, "custom_llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "llama3.1:8b")
    monkeypatch.setattr(settings, "temperature", 0.4)
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    result = generate_estimation(TRANSCRIPTION)

    assert result == EstimationResult(
        estimation="## Estimación Custom",
        model="llama3.1:8b",
        provider="custom",
        temperature=0.4,
        input_tokens=13,
        output_tokens=21,
    )
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["model"] == "llama3.1:8b"
    assert captured["temperature"] == 0.4
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1] == {"role": "user", "content": TRANSCRIPTION}
    assert ESTIMATION_EXAMPLES[0]["estimation"] in captured["messages"][0]["content"]


def test_custom_stream_yields_deltas_and_metrics(monkeypatch) -> None:
    captured: dict = {}

    class FakeStream:
        def __iter__(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="## "))])
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Estimación"))]
            )
            yield SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=13, completion_tokens=21),
            )

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeStream()

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = FakeChat()

    monkeypatch.setattr(settings, "llm_provider", "custom")
    monkeypatch.setattr(settings, "custom_llm_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(settings, "custom_llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "llama3.1:8b")
    monkeypatch.setattr(settings, "temperature", 0.2)
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    metrics = StreamMetrics(model="", provider="")
    chunks = list(stream_estimation(TRANSCRIPTION, metrics))

    assert chunks == ["## ", "Estimación"]
    assert "".join(chunks) == "## Estimación"
    assert metrics == StreamMetrics(
        model="llama3.1:8b",
        provider="custom",
        input_tokens=13,
        output_tokens=21,
    )
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["model"] == "llama3.1:8b"
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert captured["messages"][1] == {"role": "user", "content": TRANSCRIPTION}
    assert ESTIMATION_EXAMPLES[0]["estimation"] in captured["messages"][0]["content"]
