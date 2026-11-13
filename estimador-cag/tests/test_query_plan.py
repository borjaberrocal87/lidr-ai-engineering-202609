"""Plan de consulta: heurística y degradación segura."""

from __future__ import annotations

import pytest

from app.generation.rag.retrieval.query_plan import (
    DECOMPOSE,
    DIRECT,
    EXPAND,
    QueryTransformation,
    SubQuery,
    choose_technique,
    transform_query,
)


class FakeWrapper:
    def __init__(self, *, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    def complete_structured(self, **_kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result, {}


def _transformation(*queries: str) -> QueryTransformation:
    return QueryTransformation(
        subqueries=[
            SubQuery(topic=f"tema-{index}", query=query)
            for index, query in enumerate(queries)
        ]
    )


def test_short_query_is_direct() -> None:
    assert choose_technique("OAuth") == DIRECT


def test_medium_query_is_expanded() -> None:
    assert (
        choose_technique("design a resilient billing service for enterprise subscriptions")
        == EXPAND
    )


def test_long_or_multitopic_query_is_decomposed() -> None:
    assert (
        choose_technique(
            "How do we integrate Stripe Connect checkout and SAP IDoc work orders "
            "and also the OAuth authentication runbook?"
        )
        == DECOMPOSE
    )


@pytest.mark.asyncio
async def test_disabled_transform_returns_the_original_query() -> None:
    wrapper = FakeWrapper(error=AssertionError("no debe llamar al LLM"))

    plan = await transform_query(
        "design a resilient billing service for enterprise subscriptions",
        enabled=False,
        llm_wrapper=wrapper,
    )

    assert plan.technique == DIRECT
    assert plan.subqueries[0].query.startswith("design a resilient")


@pytest.mark.asyncio
async def test_expansion_uses_the_llm_and_caps_subqueries() -> None:
    wrapper = FakeWrapper(
        result=_transformation(
            "billing service for subscriptions",
            "recurring payments platform",
            "subscription management backend",
            "enterprise billing API",
        )
    )

    plan = await transform_query(
        "design a resilient billing service for enterprise subscriptions",
        llm_wrapper=wrapper,
        max_subqueries=2,
    )

    assert plan.technique == EXPAND
    assert len(plan.subqueries) == 2


@pytest.mark.asyncio
async def test_transformer_failure_degrades_to_direct() -> None:
    plan = await transform_query(
        "design a resilient billing service for enterprise subscriptions",
        llm_wrapper=FakeWrapper(error=RuntimeError("LLM down")),
    )

    assert plan.technique == DIRECT
    assert len(plan.subqueries) == 1
