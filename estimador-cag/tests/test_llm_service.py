import re
from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs

from app.config import settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.services.llm_service import (
    DATA_BOUNDARY_INSTRUCTION,
    EstimationResult,
    LLMConfigurationError,
    LLMInputError,
    LLMProviderError,
    StreamMetrics,
    build_system_prompt,
    generate_estimation,
    stream_estimation,
)

TRANSCRIPTION = "El cliente necesita una landing page con integración con HubSpot."


def _assert_wrapped(content: str, transcription: str) -> None:
    """Comprueba que la transcripción va delimitada por una etiqueta con nonce."""
    match = re.fullmatch(
        r"<(?P<tag>transcripcion-[0-9a-f]{16})>\n(?P<body>.*)\n</(?P=tag)>",
        content,
        re.DOTALL,
    )
    assert match is not None, content
    assert match.group("body") == transcription


def test_build_system_prompt_injects_all_examples() -> None:
    prompt = build_system_prompt()

    for example in ESTIMATION_EXAMPLES:
        assert example["meeting_summary"] in prompt
        assert example["estimation"] in prompt

    assert prompt.count("### Ejemplo") == len(ESTIMATION_EXAMPLES)


def test_build_system_prompt_instruye_sobre_el_limite_de_datos() -> None:
    assert DATA_BOUNDARY_INSTRUCTION in build_system_prompt()


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
            return SimpleNamespace(
                output_text="## Estimación OpenAI",
                usage=usage,
                status="completed",
                incomplete_details=None,
            )

    class FakeOpenAI:
        def __init__(self, api_key=None, *, timeout=None, max_retries=None) -> None:
            captured["api_key"] = api_key
            captured["timeout"] = timeout
            captured["max_retries"] = max_retries
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
    assert result.truncated is False
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["temperature"] == 0.3
    assert captured["timeout"] == settings.llm_timeout_seconds
    assert captured["max_retries"] == settings.llm_max_retries
    assert captured["max_output_tokens"] == settings.llm_max_tokens
    _assert_wrapped(captured["input"], TRANSCRIPTION)
    assert ESTIMATION_EXAMPLES[0]["estimation"] in captured["instructions"]


def test_openai_detecta_respuesta_truncada(monkeypatch) -> None:
    class FakeResponses:
        def create(self, **kwargs):
            usage = SimpleNamespace(input_tokens=11, output_tokens=settings.llm_max_tokens)
            return SimpleNamespace(
                output_text="## Estimación incompleta",
                usage=usage,
                status="incomplete",
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            )

    class FakeOpenAI:
        def __init__(self, api_key=None, *, timeout=None, max_retries=None) -> None:
            self.responses = FakeResponses()

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    result = generate_estimation(TRANSCRIPTION)

    assert result.truncated is True


def test_openai_respuesta_vacia_raises(monkeypatch) -> None:
    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text="",
                usage=None,
                status="completed",
                incomplete_details=None,
            )

    class FakeOpenAI:
        def __init__(self, api_key=None, *, timeout=None, max_retries=None) -> None:
            self.responses = FakeResponses()

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    with pytest.raises(LLMProviderError, match="vacía"):
        generate_estimation(TRANSCRIPTION)


def test_openai_error_inesperado_no_se_enmascara(monkeypatch) -> None:
    class FakeResponses:
        def create(self, **kwargs):
            raise RuntimeError("conexión rota")

    class FakeOpenAI:
        def __init__(self, api_key=None, *, timeout=None, max_retries=None) -> None:
            self.responses = FakeResponses()

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    # Un error que no es de la jerarquía del SDK NO se enmascara como provider error.
    with pytest.raises(RuntimeError, match="conexión rota"):
        generate_estimation(TRANSCRIPTION)


def test_openai_error_del_sdk_es_provider_error(monkeypatch) -> None:
    from openai import APIConnectionError

    class FakeResponses:
        def create(self, **kwargs):
            raise APIConnectionError(request=None)  # type: ignore[arg-type]

    class FakeOpenAI:
        def __init__(self, api_key=None, *, timeout=None, max_retries=None) -> None:
            self.responses = FakeResponses()

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    with pytest.raises(LLMProviderError):
        generate_estimation(TRANSCRIPTION)


def test_anthropic_provider_maps_response(monkeypatch) -> None:
    captured: dict = {}

    class FakeMessages:
        def create(self, *, model, max_tokens, system, messages):
            captured.update(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            block = SimpleNamespace(type="text", text="## Estimación Anthropic")
            usage = SimpleNamespace(input_tokens=7, output_tokens=9)
            return SimpleNamespace(content=[block], usage=usage, stop_reason="end_turn")

    class FakeAnthropic:
        def __init__(self, api_key=None, *, timeout=None, max_retries=None) -> None:
            captured["api_key"] = api_key
            captured["timeout"] = timeout
            captured["max_retries"] = max_retries
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
    assert result.truncated is False
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "claude-haiku-4-5"
    assert captured["max_tokens"] == settings.llm_max_tokens
    assert captured["timeout"] == settings.llm_timeout_seconds
    assert captured["max_retries"] == settings.llm_max_retries
    _assert_wrapped(captured["messages"][0]["content"], TRANSCRIPTION)
    assert captured["messages"][0]["role"] == "user"
    assert ESTIMATION_EXAMPLES[0]["meeting_summary"] in captured["system"]


def test_anthropic_detecta_respuesta_truncada(monkeypatch) -> None:
    class FakeMessages:
        def create(self, **kwargs):
            block = SimpleNamespace(type="text", text="## Estimación incompleta")
            usage = SimpleNamespace(input_tokens=7, output_tokens=settings.llm_max_tokens)
            return SimpleNamespace(content=[block], usage=usage, stop_reason="max_tokens")

    class FakeAnthropic:
        def __init__(self, api_key=None, *, timeout=None, max_retries=None) -> None:
            self.messages = FakeMessages()

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "claude-haiku-4-5")
    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)

    result = generate_estimation(TRANSCRIPTION)

    assert result.truncated is True


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
            return SimpleNamespace(
                usage=usage,
                status="completed",
                incomplete_details=None,
            )

    class FakeResponses:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return FakeOpenAIStream()

    class FakeOpenAI:
        def __init__(self, api_key=None, *, timeout=None, max_retries=None) -> None:
            captured["api_key"] = api_key
            captured["timeout"] = timeout
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
    assert captured["timeout"] == settings.llm_timeout_seconds
    assert captured["model"] == "gpt-4o-mini"
    assert captured["max_output_tokens"] == settings.llm_max_tokens
    _assert_wrapped(captured["input"], TRANSCRIPTION)
    assert ESTIMATION_EXAMPLES[0]["estimation"] in captured["instructions"]


def test_openai_stream_detecta_truncamiento(monkeypatch) -> None:
    class FakeOpenAIStream:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> bool:
            return False

        def __iter__(self):
            yield SimpleNamespace(type="response.output_text.delta", delta="## ")

        def get_final_response(self):
            return SimpleNamespace(
                usage=None,
                status="incomplete",
                incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            )

    class FakeResponses:
        def stream(self, **kwargs):
            return FakeOpenAIStream()

    class FakeOpenAI:
        def __init__(self, api_key=None, *, timeout=None, max_retries=None) -> None:
            self.responses = FakeResponses()

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    metrics = StreamMetrics(model="", provider="")
    list(stream_estimation(TRANSCRIPTION, metrics))

    assert metrics.truncated is True


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
            return SimpleNamespace(usage=usage, stop_reason="end_turn")

    class FakeMessages:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return FakeAnthropicStream()

    class FakeAnthropic:
        def __init__(self, api_key=None, *, timeout=None, max_retries=None) -> None:
            captured["api_key"] = api_key
            captured["timeout"] = timeout
            self.messages = FakeMessages()

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "claude-haiku-4-5")
    monkeypatch.setattr(settings, "temperature", 0.2)
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
    assert captured["timeout"] == settings.llm_timeout_seconds
    assert captured["model"] == "claude-haiku-4-5"
    assert captured["max_tokens"] == settings.llm_max_tokens
    _assert_wrapped(captured["messages"][0]["content"], TRANSCRIPTION)
    assert ESTIMATION_EXAMPLES[0]["meeting_summary"] in captured["system"]


def test_anthropic_stream_detecta_truncamiento(monkeypatch) -> None:
    class FakeAnthropicStream:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> bool:
            return False

        def __stream_text__(self):
            yield "## "

        def get_final_message(self):
            return SimpleNamespace(usage=None, stop_reason="max_tokens")

    class FakeMessages:
        def stream(self, **kwargs):
            return FakeAnthropicStream()

    class FakeAnthropic:
        def __init__(self, api_key=None, *, timeout=None, max_retries=None) -> None:
            self.messages = FakeMessages()

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "claude-haiku-4-5")
    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)

    metrics = StreamMetrics(model="", provider="")
    list(stream_estimation(TRANSCRIPTION, metrics))

    assert metrics.truncated is True


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
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice], usage=usage)

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, *, timeout=None, max_retries=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["timeout"] = timeout
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
    assert result.truncated is False
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["timeout"] == settings.llm_timeout_seconds
    assert captured["model"] == "llama3.1:8b"
    assert captured["temperature"] == 0.4
    assert captured["max_tokens"] == settings.llm_max_tokens
    assert captured["messages"][0]["role"] == "system"
    _assert_wrapped(captured["messages"][1]["content"], TRANSCRIPTION)
    assert ESTIMATION_EXAMPLES[0]["estimation"] in captured["messages"][0]["content"]


def test_custom_detecta_respuesta_truncada(monkeypatch) -> None:
    class FakeCompletions:
        def create(self, **kwargs):
            message = SimpleNamespace(content="## Estimación incompleta")
            usage = SimpleNamespace(prompt_tokens=13, completion_tokens=settings.llm_max_tokens)
            choice = SimpleNamespace(message=message, finish_reason="length")
            return SimpleNamespace(choices=[choice], usage=usage)

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, *, timeout=None, max_retries=None):
            self.chat = FakeChat()

    monkeypatch.setattr(settings, "llm_provider", "custom")
    monkeypatch.setattr(settings, "custom_llm_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(settings, "custom_llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "llama3.1:8b")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    result = generate_estimation(TRANSCRIPTION)

    assert result.truncated is True


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
        def __init__(self, api_key=None, base_url=None, *, timeout=None, max_retries=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["timeout"] = timeout
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
    assert captured["timeout"] == settings.llm_timeout_seconds
    assert captured["model"] == "llama3.1:8b"
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert captured["max_tokens"] == settings.llm_max_tokens
    _assert_wrapped(captured["messages"][1]["content"], TRANSCRIPTION)
    assert ESTIMATION_EXAMPLES[0]["estimation"] in captured["messages"][0]["content"]


def test_custom_stream_detecta_truncamiento(monkeypatch) -> None:
    class FakeStream:
        def __iter__(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="## "))])
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=None, finish_reason="length")],
            )

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeStream()

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, *, timeout=None, max_retries=None):
            self.chat = FakeChat()

    monkeypatch.setattr(settings, "llm_provider", "custom")
    monkeypatch.setattr(settings, "custom_llm_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(settings, "custom_llm_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "llama3.1:8b")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    metrics = StreamMetrics(model="", provider="")
    list(stream_estimation(TRANSCRIPTION, metrics))

    assert metrics.truncated is True


def test_openai_no_marca_truncado_por_content_filter(monkeypatch) -> None:
    class FakeResponses:
        def create(self, **kwargs):
            usage = SimpleNamespace(input_tokens=11, output_tokens=5)
            return SimpleNamespace(
                output_text="## Estimación filtrada",
                usage=usage,
                status="incomplete",
                incomplete_details=SimpleNamespace(reason="content_filter"),
            )

    class FakeOpenAI:
        def __init__(self, api_key=None, *, timeout=None, max_retries=None) -> None:
            self.responses = FakeResponses()

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    result = generate_estimation(TRANSCRIPTION)

    # `incomplete` por filtro de contenido no es un truncamiento por tokens.
    assert result.truncated is False


def test_transcription_demasiado_larga_raises_input_error(monkeypatch) -> None:
    monkeypatch.setattr(settings, "transcription_max_length", 20)

    with pytest.raises(LLMInputError, match="entre"):
        generate_estimation(TRANSCRIPTION)


def test_transcription_demasiado_corta_raises_input_error(monkeypatch) -> None:
    monkeypatch.setattr(settings, "transcription_min_length", 100)

    with pytest.raises(LLMInputError, match="entre"):
        generate_estimation(TRANSCRIPTION)


def test_stream_respuesta_vacia_raises(monkeypatch) -> None:
    class FakeOpenAIStream:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> bool:
            return False

        def __iter__(self):
            yield SimpleNamespace(type="response.completed")

        def get_final_response(self):
            return SimpleNamespace(usage=None, status="completed", incomplete_details=None)

    class FakeResponses:
        def stream(self, **kwargs):
            return FakeOpenAIStream()

    class FakeOpenAI:
        def __init__(self, api_key=None, *, timeout=None, max_retries=None) -> None:
            self.responses = FakeResponses()

    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "gpt-4o-mini")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    with pytest.raises(LLMProviderError, match="vacía"):
        list(stream_estimation(TRANSCRIPTION))


def test_anthropic_temperature_ignorada_emite_warning(monkeypatch) -> None:
    class FakeMessages:
        def create(self, **kwargs):
            block = SimpleNamespace(type="text", text="## Estimación")
            usage = SimpleNamespace(input_tokens=1, output_tokens=2)
            return SimpleNamespace(content=[block], usage=usage, stop_reason="end_turn")

    class FakeAnthropic:
        def __init__(self, api_key=None, *, timeout=None, max_retries=None) -> None:
            self.messages = FakeMessages()

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_model", "claude-haiku-4-5")
    monkeypatch.setattr(settings, "temperature", 0.9)
    monkeypatch.setattr("anthropic.Anthropic", FakeAnthropic)
    monkeypatch.setattr("app.services.llm_service._temperature_warning_emitted", False)

    with capture_logs() as logs:
        generate_estimation(TRANSCRIPTION)

    assert "llm.temperature.ignored" in [entry["event"] for entry in logs]


def test_build_system_prompt_acepta_ejemplos_externos() -> None:
    ejemplos = [{"meeting_summary": "resumen controlado", "estimation": "estimación controlada"}]

    prompt = build_system_prompt(ejemplos)

    assert "resumen controlado" in prompt
    assert "estimación controlada" in prompt
    assert ESTIMATION_EXAMPLES[0]["estimation"] not in prompt


def test_stream_transcription_fuera_de_rango_raises_input_error(monkeypatch) -> None:
    monkeypatch.setattr(settings, "transcription_max_length", 20)

    with pytest.raises(LLMInputError, match="entre"):
        stream_estimation(TRANSCRIPTION)


def test_system_prompt_instruye_contra_el_anclaje_de_cifras() -> None:
    assert "no copies sus números" in build_system_prompt()
