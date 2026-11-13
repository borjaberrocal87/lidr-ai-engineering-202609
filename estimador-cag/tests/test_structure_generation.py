"""Estructura interactiva: settings propios y deadline duro (S10)."""

from __future__ import annotations

import time

import pytest

from app.config import settings
from app.generation.rag.errors import RagGenerationTimeoutError
from app.generation.rag.grounded_estimator import generate_structure
from app.generation.rag.schemas import GroundedEstimate, ProjectBrief


def _brief() -> ProjectBrief:
    return ProjectBrief(functionality="Claims portal with OAuth")


def _estimate() -> GroundedEstimate:
    return GroundedEstimate(
        total_engineer_days=None,
        modules=[],
        duration_weeks=None,
        sources=[],
        assumptions=[],
        confidence="medium",
        reasoning="estructura de prueba",
        insufficient_context_explanation=None,
    )


class RecordingWrapper:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.kwargs: dict = {}

    def complete_structured(self, **kwargs):
        self.kwargs = kwargs
        if self.delay:
            time.sleep(self.delay)
        return _estimate(), {}


@pytest.mark.asyncio
async def test_structure_uses_its_own_fast_settings(monkeypatch) -> None:
    wrapper = RecordingWrapper()
    monkeypatch.setattr("app.dependencies.get_llm_wrapper", lambda: wrapper)
    monkeypatch.setattr(settings, "rag_structure_model", "gpt-structure-test")
    monkeypatch.setattr(settings, "rag_structure_reasoning_effort", "low")
    monkeypatch.setattr(settings, "rag_structure_max_tokens", 1234)
    monkeypatch.setattr(settings, "rag_structure_timeout_seconds", 5.0)

    estimate = await generate_structure(_brief())

    assert estimate.confidence == "medium"
    assert wrapper.kwargs["model_override"] == "gpt-structure-test"
    assert wrapper.kwargs["reasoning_effort"] == "low"
    assert wrapper.kwargs["max_tokens"] == 1234
    assert wrapper.kwargs["max_retries"] == 0
    assert wrapper.kwargs["num_retries"] == 0
    assert "temperature" in wrapper.kwargs


@pytest.mark.asyncio
async def test_structure_raises_timeout_instead_of_hanging(monkeypatch) -> None:
    wrapper = RecordingWrapper(delay=0.2)
    monkeypatch.setattr("app.dependencies.get_llm_wrapper", lambda: wrapper)
    monkeypatch.setattr(settings, "rag_structure_timeout_seconds", 0.01)

    with pytest.raises(RagGenerationTimeoutError):
        await generate_structure(_brief())
