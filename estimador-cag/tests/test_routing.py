"""Enrutado de fuentes en cascada (explícito → reglas → clasificador → fallback)."""

from __future__ import annotations

import pytest

from app.generation.rag.retrieval.routing import (
    CLASSIFIER,
    EXPLICIT,
    FALLBACK,
    RULES,
    RouteClassification,
    route,
)
from app.generation.rag.retrieval.sources import Collection


class FakeWrapper:
    def __init__(self, *, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error

    def complete_structured(self, *, temperature, **_kwargs):
        if self.error is not None:
            raise self.error
        return self.result, {}


@pytest.mark.asyncio
async def test_explicit_collections_skip_the_classifier() -> None:
    decision = await route(
        "cualquier cosa",
        explicit=[Collection.TRANSCRIPT, Collection.BUDGET],
        llm_wrapper=FakeWrapper(error=AssertionError("no debe clasificar")),
    )

    assert decision.level == EXPLICIT
    assert decision.targets == [Collection.TRANSCRIPT, Collection.BUDGET]


@pytest.mark.asyncio
async def test_vocabulary_rules_win_before_the_llm() -> None:
    decision = await route(
        "¿cuánto coste y horas tiene este presupuesto?",
        llm_wrapper=FakeWrapper(error=AssertionError("no debe clasificar")),
    )

    assert decision.level == RULES
    assert decision.targets == [Collection.BUDGET]


@pytest.mark.asyncio
async def test_classifier_is_used_when_no_rule_fires() -> None:
    wrapper = FakeWrapper(
        result=RouteClassification(
            targets=[Collection.TECHNICAL_DOC, Collection.TRANSCRIPT],
            reason="La pregunta pide referencia y contexto de reunión.",
        )
    )

    decision = await route("Help me choose a direction", llm_wrapper=wrapper)

    assert decision.level == CLASSIFIER
    assert decision.targets == [Collection.TECHNICAL_DOC, Collection.TRANSCRIPT]


@pytest.mark.asyncio
async def test_classifier_failure_degrades_to_all_collections() -> None:
    decision = await route(
        "Help me choose a direction",
        llm_wrapper=FakeWrapper(error=RuntimeError("LLM down")),
    )

    assert decision.level == FALLBACK
    assert set(decision.targets) == {
        Collection.BUDGET,
        Collection.TRANSCRIPT,
        Collection.TECHNICAL_DOC,
    }


@pytest.mark.asyncio
async def test_max_targets_trims_explicit_and_rule_results() -> None:
    decision = await route(
        "presupuesto reunión runbook",
        max_targets=1,
    )

    assert len(decision.targets) == 1
