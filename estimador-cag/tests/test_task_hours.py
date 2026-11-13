"""Consenso de horas por tarea y su contrato de no-match."""

from __future__ import annotations

import pytest

from app.generation.rag.schemas import (
    RetrievalOutcome,
    RetrievedChunk,
    TaskHoursModuleInput,
    TaskHoursTaskInput,
)
from app.generation.rag.task_hours import (
    _consensus,
    compose_task_search_text,
    estimate_all,
    estimate_one,
)


class FakeEmbedder:
    def embed_one(self, _text: str) -> list[float]:
        return [0.1, 0.2]


def _chunk(chunk_id: int, hours: int | None, *, distance: float = 0.2) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id=1,
        content=f"historical task {chunk_id}",
        chunk_type="historical_task",
        distance=distance,
        budget_id="TASK-2022-0001",
        estimated_hours=hours,
    )


def test_search_text_mirrors_the_historical_chunk_shape() -> None:
    text = compose_task_search_text("Auth", "OAuth2 login", "PKCE flow")

    assert text == "Module: Auth\nTask: OAuth2 login\nPKCE flow"


def test_consensus_rewards_close_and_agreeing_neighbours() -> None:
    hours, reliability, dispersion = _consensus([(40, 0.10), (42, 0.12), (41, 0.11)])

    assert hours == 41
    assert reliability > 0.8
    assert dispersion < 0.1


def test_consensus_penalises_disagreement() -> None:
    _hours, reliability, dispersion = _consensus([(10, 0.1), (90, 0.1)])

    assert dispersion > 0.5
    assert reliability < 0.5


@pytest.mark.asyncio
async def test_estimate_one_returns_neighbour_evidence(monkeypatch) -> None:
    async def fake_retrieve(**_kwargs):
        return RetrievalOutcome(
            chunks=[_chunk(1, 40), _chunk(2, 42)],
            low_confidence=False,
            candidates_evaluated=2,
        )

    monkeypatch.setattr("app.generation.rag.task_hours.retrieve", fake_retrieve)

    estimate = await estimate_one(
        "Auth",
        "OAuth2 login",
        "PKCE flow",
        store=object(),
        session_factory=object,
        embedder=FakeEmbedder(),
    )

    assert estimate.has_match is True
    assert estimate.estimated_hours == 41
    assert len(estimate.neighbors) == 2
    assert estimate.neighbors[0].budget_id == "TASK-2022-0001"


@pytest.mark.asyncio
async def test_estimate_one_flags_tasks_without_hours(monkeypatch) -> None:
    async def fake_retrieve(**_kwargs):
        return RetrievalOutcome(
            chunks=[_chunk(1, None)],
            low_confidence=False,
            candidates_evaluated=1,
        )

    monkeypatch.setattr("app.generation.rag.task_hours.retrieve", fake_retrieve)

    estimate = await estimate_one(
        "Auth",
        "OAuth2 login",
        None,
        store=object(),
        session_factory=object,
        embedder=FakeEmbedder(),
    )

    assert estimate.has_match is False
    assert estimate.estimated_hours is None


@pytest.mark.asyncio
async def test_estimate_all_preserves_submitted_order(monkeypatch) -> None:
    async def fake_retrieve(**_kwargs):
        return RetrievalOutcome(
            chunks=[_chunk(1, 40)],
            low_confidence=False,
            candidates_evaluated=1,
        )

    monkeypatch.setattr("app.generation.rag.task_hours.retrieve", fake_retrieve)

    result = await estimate_all(
        [
            TaskHoursModuleInput(
                name="Auth",
                tasks=[TaskHoursTaskInput(name="Login"), TaskHoursTaskInput(name="MFA")],
            )
        ],
        store=object(),
        session_factory=object,
        embedder=FakeEmbedder(),
    )

    assert [task.task for task in result.tasks] == ["Login", "MFA"]
