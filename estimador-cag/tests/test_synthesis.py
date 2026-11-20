"""Síntesis de horas: el rango aparece solo cuando los vecinos se contradicen."""

from __future__ import annotations

import pytest

from app.generation.rag.quality.synthesis import (
    deterministic_reason,
    is_contradiction,
    synthesize_hour_range,
)
from app.generation.rag.schemas import HourRange, TaskNeighbor


def _neighbor(hours: int, distance: float = 0.1) -> TaskNeighbor:
    return TaskNeighbor(source_id=hours, estimated_hours=hours, distance=distance)


def test_is_contradiction_uses_threshold() -> None:
    assert is_contradiction(0.4, 0.35) is True
    assert is_contradiction(0.35, 0.35) is False
    assert is_contradiction(None, 0.35) is False


@pytest.mark.asyncio
async def test_no_range_when_neighbours_agree() -> None:
    neighbors = [_neighbor(40), _neighbor(42)]

    assert await synthesize_hour_range(neighbors, 0.05, threshold=0.35) is None
    assert await synthesize_hour_range([], 0.9, threshold=0.35) is None


@pytest.mark.asyncio
async def test_contradiction_emits_min_max_range_with_deterministic_reason() -> None:
    neighbors = [_neighbor(40), _neighbor(90), _neighbor(65)]

    result = await synthesize_hour_range(neighbors, 0.5, threshold=0.35)

    assert isinstance(result, HourRange)
    assert (result.low_hours, result.high_hours) == (40, 90)
    assert result.reason == deterministic_reason(neighbors)
    assert "40h" in result.reason and "90h" in result.reason


@pytest.mark.asyncio
async def test_rephraser_can_rewrite_the_reason() -> None:
    neighbors = [_neighbor(40), _neighbor(90)]

    async def rephrase(_neighbors: list[TaskNeighbor], _base: HourRange) -> str:
        return "la integración con el ERP no estaba en el alcance inicial"

    result = await synthesize_hour_range(neighbors, 0.5, threshold=0.35, rephrase=rephrase)

    assert result is not None
    assert result.reason == "la integración con el ERP no estaba en el alcance inicial"
    assert (result.low_hours, result.high_hours) == (40, 90)


@pytest.mark.asyncio
async def test_failing_rephraser_degrades_to_deterministic_reason() -> None:
    neighbors = [_neighbor(40), _neighbor(90)]

    async def rephrase(_neighbors: list[TaskNeighbor], _base: HourRange) -> str:
        raise RuntimeError("proveedor caído")

    result = await synthesize_hour_range(neighbors, 0.5, threshold=0.35, rephrase=rephrase)

    assert result is not None
    assert result.reason == deterministic_reason(neighbors)


@pytest.mark.asyncio
async def test_blank_rephrased_reason_keeps_deterministic_one() -> None:
    neighbors = [_neighbor(40), _neighbor(90)]

    async def rephrase(_neighbors: list[TaskNeighbor], _base: HourRange) -> str:
        return "   "

    result = await synthesize_hour_range(neighbors, 0.5, threshold=0.35, rephrase=rephrase)

    assert result is not None
    assert result.reason == deterministic_reason(neighbors)
