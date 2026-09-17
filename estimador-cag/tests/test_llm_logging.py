"""Tests del logging estructurado de las llamadas al LLM."""

from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs

from app.config import settings
from app.services.llm_service import (
    LLMConfigurationError,
    generate_estimation,
    stream_estimation,
)

TRANSCRIPTION = "El cliente necesita una landing page con integración con HubSpot."


def _patch_openai_success(monkeypatch) -> None:
    class FakeResponses:
        def create(self, **kwargs):
            usage = SimpleNamespace(input_tokens=11, output_tokens=22)
            return SimpleNamespace(output_text="## Estimación", usage=usage)

    class FakeOpenAI:
        def __init__(self, api_key=None) -> None:
            self.responses = FakeResponses()

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "temperature", 0.3)
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)


def _patch_openai_stream(monkeypatch, *, error: str | None = None) -> None:
    class FakeOpenAIStream:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> bool:
            return False

        def __iter__(self):
            if error is not None:
                raise RuntimeError(error)
            yield SimpleNamespace(type="response.output_text.delta", delta="## ")
            yield SimpleNamespace(type="response.output_text.delta", delta="Estimación")

        def get_final_response(self):
            usage = SimpleNamespace(input_tokens=5, output_tokens=7)
            return SimpleNamespace(usage=usage)

    class FakeResponses:
        def stream(self, **kwargs):
            return FakeOpenAIStream()

    class FakeOpenAI:
        def __init__(self, api_key=None) -> None:
            self.responses = FakeResponses()

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr(settings, "temperature", 0.2)
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)


def _events(logs: list[dict]) -> list[str]:
    return [entry["event"] for entry in logs]


def test_generate_estimation_logs_start_and_end(monkeypatch) -> None:
    _patch_openai_success(monkeypatch)

    with capture_logs() as logs:
        result = generate_estimation(TRANSCRIPTION)

    assert result.estimation == "## Estimación"
    assert _events(logs) == ["llm.call.start", "llm.call.end"]

    start, end = logs
    assert start["provider"] == "openai"
    assert start["model"] == "gpt-4o-mini"
    assert start["temperature"] == 0.3
    assert start["transcription_chars"] == len(TRANSCRIPTION)
    assert end["input_tokens"] == 11
    assert end["output_tokens"] == 22
    assert isinstance(end["latency_ms"], float)


def test_generate_estimation_logs_error(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "")

    with capture_logs() as logs, pytest.raises(LLMConfigurationError):
        generate_estimation(TRANSCRIPTION)

    events = _events(logs)
    assert "llm.call.start" in events
    assert "llm.call.error" in events
    assert "llm.call.end" not in events
    error = next(entry for entry in logs if entry["event"] == "llm.call.error")
    assert isinstance(error["latency_ms"], float)


def test_stream_estimation_logs_start_and_end(monkeypatch) -> None:
    _patch_openai_stream(monkeypatch)

    with capture_logs() as logs:
        chunks = list(stream_estimation(TRANSCRIPTION))

    assert chunks == ["## ", "Estimación"]
    events = _events(logs)
    assert events.count("llm.stream.start") == 1
    assert "llm.stream.end" in events

    end = next(entry for entry in logs if entry["event"] == "llm.stream.end")
    assert end["input_tokens"] == 5
    assert end["output_tokens"] == 7
    assert isinstance(end["latency_ms"], float)


def test_stream_estimation_logs_error_during_consumption(monkeypatch) -> None:
    _patch_openai_stream(monkeypatch, error="boom")

    with capture_logs() as logs, pytest.raises(RuntimeError, match="boom"):
        list(stream_estimation(TRANSCRIPTION))

    events = _events(logs)
    assert "llm.stream.start" in events
    assert "llm.stream.error" in events
    assert "llm.stream.end" not in events


def test_stream_estimation_logs_eager_config_error(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "")

    with capture_logs() as logs, pytest.raises(LLMConfigurationError):
        stream_estimation(TRANSCRIPTION)

    events = _events(logs)
    assert "llm.stream.start" in events
    assert "llm.stream.error" in events
    assert "llm.stream.end" not in events


def test_logs_do_not_leak_api_key(monkeypatch) -> None:
    secret = "super-secret-api-key"
    _patch_openai_success(monkeypatch)
    monkeypatch.setattr(settings, "open_ai_key", secret)

    with capture_logs() as logs:
        generate_estimation(TRANSCRIPTION)

    assert secret not in str(logs)


def test_custom_provider_logs_are_tagged(monkeypatch) -> None:
    class FakeCompletions:
        def create(self, **kwargs):
            message = SimpleNamespace(content="## Estimación")
            usage = SimpleNamespace(prompt_tokens=13, completion_tokens=21)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None) -> None:
            self.chat = FakeChat()

    monkeypatch.setattr(settings, "llm_provider", "custom")
    monkeypatch.setattr(settings, "custom_llm_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(settings, "custom_llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "qwen3.8-flash")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    with capture_logs() as logs:
        generate_estimation(TRANSCRIPTION)

    start = next(entry for entry in logs if entry["event"] == "llm.call.start")
    assert start["provider"] == "custom"
    assert start["model"] == "qwen3.8-flash"
