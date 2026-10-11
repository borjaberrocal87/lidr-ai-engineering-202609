"""Tests de la pipeline estructurada (``EstimationService.estimate``)."""

from typing import Any

import pytest

from app.config import settings
from app.schemas.estimations import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)
from app.services import estimation
from app.services.errors import (
    LLMConfigurationError,
    LLMProviderError,
)
from app.services.estimation import EstimationService, StructuredEstimation
from tests.conftest import FakeWrapper

DESCRIPTION = "El cliente necesita una landing page con integración con HubSpot."


def _request(**overrides: Any) -> EstimationRequest:
    base: dict[str, Any] = {
        "description": DESCRIPTION,
        "project_type": ProjectType.WEB_SAAS,
        "detail_level": DetailLevel.MEDIUM,
        "output_format": OutputFormat.PHASES_TABLE,
    }
    base.update(overrides)
    return EstimationRequest(**base)


def _service() -> EstimationService:
    return EstimationService()


def _use_wrapper(monkeypatch, fake: FakeWrapper) -> None:
    monkeypatch.setattr(estimation, "get_llm_wrapper", lambda: fake)


def test_generate_structured_estimation_maps_result(monkeypatch) -> None:
    fake = FakeWrapper()
    _use_wrapper(monkeypatch, fake)

    outcome = _service().estimate(_request())

    assert isinstance(outcome, StructuredEstimation)
    assert outcome.result.total_cost_eur == 2000
    assert outcome.model == "gpt-4o-mini"
    assert outcome.provider == "openai"
    assert outcome.input_tokens == 11
    assert outcome.cache_hit is False
    call = fake.structured_calls[0]
    assert "<examples>" in call["system_prompt"]
    assert DESCRIPTION in call["user_message"]
    assert call["max_retries"] == settings.structured_max_retries


def test_generate_structured_estimation_reuses_cache(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "test-key")
    monkeypatch.setattr(settings, "temperature", 0.2)
    fake = FakeWrapper()
    _use_wrapper(monkeypatch, fake)

    service = _service()
    first = service.estimate(_request())
    second = service.estimate(_request())

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(fake.structured_calls) == 1
    assert second.result.total_cost_eur == 2000


def test_generate_structured_estimation_missing_key_raises(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "open_ai_key", "")

    with pytest.raises(LLMConfigurationError, match="OPEN_AI_KEY"):
        _service().estimate(_request())


def test_generate_structured_estimation_propagates_provider_error(monkeypatch) -> None:
    fake = FakeWrapper(error=LLMProviderError("Fallo del proveedor LLM 'openai'."))
    _use_wrapper(monkeypatch, fake)

    with pytest.raises(LLMProviderError):
        _service().estimate(_request())
