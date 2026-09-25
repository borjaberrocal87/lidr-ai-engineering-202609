"""Tests del servicio de estimación con el wrapper de proveedores mockeado."""

import re
from types import SimpleNamespace
from typing import Any

import pytest
from structlog.testing import capture_logs

from app.context.examples import ESTIMATION_EXAMPLES
from app.services import llm_service
from app.services.cache import InMemoryCache
from app.services.errors import (
    LLMConfigurationError,
    LLMInputError,
    LLMProviderError,
)
from app.services.llm_service import (
    DATA_BOUNDARY_INSTRUCTION,
    EstimationResult,
    StreamMetrics,
    build_system_prompt,
    generate_estimation,
    stream_estimation,
)
from app.services.llm_wrapper import LLMWrapper
from tests.conftest import FakeWrapper

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


def _use_wrapper(monkeypatch, fake: FakeWrapper) -> None:
    monkeypatch.setattr(llm_service, "get_llm_wrapper", lambda: fake)


def test_build_system_prompt_injects_all_examples() -> None:
    prompt = build_system_prompt()

    for example in ESTIMATION_EXAMPLES:
        assert example["meeting_summary"] in prompt
        assert example["estimation"] in prompt

    assert prompt.count("### Ejemplo") == len(ESTIMATION_EXAMPLES)


def test_build_system_prompt_instruye_sobre_el_limite_de_datos() -> None:
    assert DATA_BOUNDARY_INSTRUCTION in build_system_prompt()


def test_build_system_prompt_acepta_ejemplos_externos() -> None:
    ejemplos = [{"meeting_summary": "resumen controlado", "estimation": "estimación controlada"}]

    prompt = build_system_prompt(ejemplos)

    assert "resumen controlado" in prompt
    assert "estimación controlada" in prompt
    assert ESTIMATION_EXAMPLES[0]["estimation"] not in prompt


def test_system_prompt_instruye_contra_el_anclaje_de_cifras() -> None:
    assert "no copies sus números" in build_system_prompt()


def test_unsupported_provider_raises(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "llm_provider", "gemini")

    with pytest.raises(LLMConfigurationError):
        generate_estimation(TRANSCRIPTION)


def test_openai_missing_key_raises(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "llm_provider", "openai")
    monkeypatch.setattr(llm_service.settings, "open_ai_key", "")

    with pytest.raises(LLMConfigurationError, match="OPEN_AI_KEY"):
        generate_estimation(TRANSCRIPTION)


def test_anthropic_missing_key_raises(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "llm_provider", "anthropic")
    monkeypatch.setattr(llm_service.settings, "anthropic_api_key", "")

    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        generate_estimation(TRANSCRIPTION)


def test_custom_missing_base_url_raises(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "llm_provider", "custom")
    monkeypatch.setattr(llm_service.settings, "custom_llm_base_url", "")
    monkeypatch.setattr(llm_service.settings, "custom_llm_api_key", "test-key")

    with pytest.raises(LLMConfigurationError, match="CUSTOM_LLM_BASE_URL"):
        generate_estimation(TRANSCRIPTION)


def test_custom_missing_key_raises(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "llm_provider", "custom")
    monkeypatch.setattr(llm_service.settings, "custom_llm_base_url", "http://localhost:11434/v1")
    monkeypatch.setattr(llm_service.settings, "custom_llm_api_key", "")

    with pytest.raises(LLMConfigurationError, match="CUSTOM_LLM_API_KEY"):
        generate_estimation(TRANSCRIPTION)


def test_transcription_demasiado_larga_raises_input_error(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "transcription_max_length", 20)

    with pytest.raises(LLMInputError, match="entre"):
        generate_estimation(TRANSCRIPTION)


def test_transcription_demasiado_corta_raises_input_error(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "transcription_min_length", 100)

    with pytest.raises(LLMInputError, match="entre"):
        generate_estimation(TRANSCRIPTION)


def test_stream_transcription_fuera_de_rango_raises_input_error(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "transcription_max_length", 20)

    with pytest.raises(LLMInputError, match="entre"):
        stream_estimation(TRANSCRIPTION)


def test_generate_estimation_maps_wrapper_result(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "llm_provider", "openai")
    monkeypatch.setattr(llm_service.settings, "temperature", 0.3)
    fake = FakeWrapper(
        result={
            "estimation": "## Estimación OpenAI",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "input_tokens": 11,
            "output_tokens": 22,
            "truncated": False,
            "latency_ms": 10.0,
            "cost_usd": 0.00042,
            "fallback_used": True,
            "cache_hit": True,
        }
    )
    _use_wrapper(monkeypatch, fake)

    result = generate_estimation(TRANSCRIPTION)

    assert result == EstimationResult(
        estimation="## Estimación OpenAI",
        model="gpt-4o-mini",
        provider="openai",
        temperature=0.3,
        input_tokens=11,
        output_tokens=22,
        truncated=False,
        cache_hit=True,
        cost_usd=0.00042,
        fallback_used=True,
    )


def test_generate_estimation_envia_prompt_cag_y_transcripcion_delimitada(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "temperature", 0.2)
    fake = FakeWrapper()
    _use_wrapper(monkeypatch, fake)

    generate_estimation(TRANSCRIPTION)

    call = fake.complete_calls[0]
    assert ESTIMATION_EXAMPLES[0]["estimation"] in call["system_prompt"]
    assert DATA_BOUNDARY_INSTRUCTION in call["system_prompt"]
    _assert_wrapped(call["user_message"], TRANSCRIPTION)
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == llm_service.settings.llm_max_tokens


def test_generate_estimation_respuesta_vacia_raises(monkeypatch) -> None:
    fake = FakeWrapper(result={**FakeWrapper().result, "estimation": ""})
    _use_wrapper(monkeypatch, fake)

    with pytest.raises(LLMProviderError, match="vacía"):
        generate_estimation(TRANSCRIPTION)


def test_generate_estimation_propaga_error_del_wrapper(monkeypatch) -> None:
    fake = FakeWrapper(error=LLMProviderError("Fallo del proveedor LLM 'openai'."))
    _use_wrapper(monkeypatch, fake)

    with pytest.raises(LLMProviderError):
        generate_estimation(TRANSCRIPTION)


def test_stream_estimation_yields_deltas_and_metrics(monkeypatch) -> None:
    fake = FakeWrapper(chunks=["## ", "Estimación"])
    _use_wrapper(monkeypatch, fake)

    metrics = StreamMetrics()
    chunks = list(stream_estimation(TRANSCRIPTION, metrics))

    assert chunks == ["## ", "Estimación"]
    assert metrics == StreamMetrics(
        model="gpt-4o-mini", provider="openai", input_tokens=11, output_tokens=22
    )
    call = fake.stream_calls[0]
    assert ESTIMATION_EXAMPLES[0]["estimation"] in call["system_prompt"]
    _assert_wrapped(call["user_message"], TRANSCRIPTION)
    assert call["max_tokens"] == llm_service.settings.llm_max_tokens


def test_stream_respuesta_vacia_raises(monkeypatch) -> None:
    fake = FakeWrapper(chunks=[])
    _use_wrapper(monkeypatch, fake)

    with pytest.raises(LLMProviderError, match="vacía"):
        list(stream_estimation(TRANSCRIPTION))


def test_stream_missing_key_raises_eagerly(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "llm_provider", "openai")
    monkeypatch.setattr(llm_service.settings, "open_ai_key", "")

    with pytest.raises(LLMConfigurationError, match="OPEN_AI_KEY"):
        stream_estimation(TRANSCRIPTION)


def test_stream_unsupported_provider_raises(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "llm_provider", "gemini")

    with pytest.raises(LLMConfigurationError):
        stream_estimation(TRANSCRIPTION)


def test_anthropic_temperature_ignorada_emite_warning(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "llm_provider", "anthropic")
    monkeypatch.setattr(llm_service.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(llm_service.settings, "temperature", 0.9)
    monkeypatch.setattr(llm_service, "_temperature_warning_emitted", False)
    _use_wrapper(monkeypatch, FakeWrapper())

    with capture_logs() as logs:
        result = generate_estimation(TRANSCRIPTION)

    assert "llm.temperature.ignored" in [entry["event"] for entry in logs]
    assert result.temperature is None


def _openai_settings(monkeypatch) -> None:
    monkeypatch.setattr(llm_service.settings, "llm_provider", "openai")
    monkeypatch.setattr(llm_service.settings, "open_ai_key", "test-key")
    monkeypatch.setattr(llm_service.settings, "temperature", 0.2)


def _completion(text: str = "## Estimación") -> SimpleNamespace:
    return SimpleNamespace(
        model="gpt-4o-mini",
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )


def _real_wrapper() -> LLMWrapper:
    return LLMWrapper(
        primary_model="gpt-4o-mini",
        primary_provider="openai",
        cache=InMemoryCache(ttl=60),
        openai_api_key="test-key",
    )


def test_generate_estimation_reusa_cache_con_la_misma_transcripcion(monkeypatch) -> None:
    """Regresión: el nonce de `_wrap_transcription` no debe cambiar la clave de caché."""
    _openai_settings(monkeypatch)
    calls = {"n": 0}

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        calls["n"] += 1
        return _completion()

    wrapper = _real_wrapper()
    monkeypatch.setattr(wrapper.router, "completion", fake_completion)
    monkeypatch.setattr(llm_service, "get_llm_wrapper", lambda: wrapper)

    first = generate_estimation(TRANSCRIPTION)
    second = generate_estimation(TRANSCRIPTION)

    assert calls["n"] == 1
    assert first.cache_hit is False
    assert second.cache_hit is True


def test_stream_estimation_reusa_cache_con_la_misma_transcripcion(monkeypatch) -> None:
    _openai_settings(monkeypatch)
    calls = {"n": 0}

    def fake_completion(**kwargs: Any):
        calls["n"] += 1
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="## "))])
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Estimación"))]
        )

    wrapper = _real_wrapper()
    monkeypatch.setattr(wrapper.router, "completion", fake_completion)
    monkeypatch.setattr(llm_service, "get_llm_wrapper", lambda: wrapper)

    first_metrics = StreamMetrics()
    first = list(stream_estimation(TRANSCRIPTION, first_metrics))
    second_metrics = StreamMetrics()
    second = list(stream_estimation(TRANSCRIPTION, second_metrics))

    assert calls["n"] == 1
    assert first == ["## ", "Estimación"]
    assert "".join(second) == "## Estimación"
    assert first_metrics.cache_hit is False
    assert second_metrics.cache_hit is True
