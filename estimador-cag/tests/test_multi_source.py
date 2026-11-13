"""Pipeline multifuente: fusión entre fuentes, procedencia y reranking."""

from __future__ import annotations

from datetime import date

import pytest

from app.generation.rag.retrieval.multi_source import StageConfig, retrieve_multi_source
from app.generation.rag.retrieval.sources import Collection
from app.generation.rag.store.models import BudgetChunkRow, TranscriptChunkRow
from app.generation.rag.store.repository import ChunkMatch


class FakeSession:
    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *_exc) -> bool:
        return False


def _factory() -> FakeSession:
    return FakeSession()


class FakeStore:
    search_mode = "vector"

    async def search_with_filters(self, _session, *, model, **_kwargs):
        if model is BudgetChunkRow:
            return (
                [
                    ChunkMatch(
                        chunk_id=1,
                        document_id=1,
                        chunk_type="historical_task",
                        content="OAuth2 login",
                        distance=0.1,
                        metadata={
                            "budget_id": "TASK-2022-0001",
                            "client_sector": "finance",
                            "year": 2022,
                            "estimated_hours": 40,
                        },
                    )
                ],
                1,
            )
        if model is TranscriptChunkRow:
            return (
                [
                    ChunkMatch(
                        chunk_id=1,
                        document_id=2,
                        chunk_type="meeting_segment",
                        content="The client confirmed OAuth scope.",
                        distance=0.2,
                        metadata={"transcript_id": "TR-2025-101", "meeting_date": "2025-02-18"},
                    )
                ],
                1,
            )
        return ([], 0)

    async def search_lexical(self, _session, **_kwargs):
        return []


class FakeEmbedder:
    def embed_one(self, _text: str) -> list[float]:
        return [0.1, 0.2]


class FakeReranker:
    def rerank_with_scores(self, _query, candidates, *, top_n):
        return [
            (candidate, 0.9 - index * 0.1)
            for index, candidate in enumerate(candidates[:top_n])
        ]


@pytest.mark.asyncio
async def test_explicit_sources_are_searched_and_merged_with_provenance() -> None:
    outcome = await retrieve_multi_source(
        query_text="authentication and meeting notes",
        embedder=FakeEmbedder(),
        store=FakeStore(),
        session_factory=_factory,
        stages=StageConfig(
            routing_enabled=False,
            query_plan_enabled=False,
            search_mode="vector",
            rerank=False,
            top_k=5,
            distance_threshold=2.0,
        ),
        reference_date=date(2025, 3, 1),
        explicit_collections=[Collection.BUDGET, Collection.TRANSCRIPT],
    )

    assert outcome.routing.level == "explicit"
    assert outcome.technique == "direct"
    assert [chunk.collection for chunk in outcome.chunks] == ["budget", "transcript"]
    assert outcome.cardinality == {"budget": 1, "transcript": 1}
    assert outcome.low_confidence is False


@pytest.mark.asyncio
async def test_reranking_populates_the_final_relevance_score() -> None:
    outcome = await retrieve_multi_source(
        query_text="authentication and meeting notes",
        embedder=FakeEmbedder(),
        store=FakeStore(),
        session_factory=_factory,
        stages=StageConfig(
            routing_enabled=False,
            query_plan_enabled=False,
            search_mode="vector",
            rerank=True,
            top_k=5,
            distance_threshold=2.0,
        ),
        reference_date=date(2025, 3, 1),
        reranker=FakeReranker(),
        explicit_collections=[Collection.BUDGET, Collection.TRANSCRIPT],
    )

    assert outcome.chunks
    assert all(chunk.relevance_score is not None for chunk in outcome.chunks)


@pytest.mark.asyncio
async def test_empty_sources_mark_low_confidence() -> None:
    outcome = await retrieve_multi_source(
        query_text="authentication and meeting notes",
        embedder=FakeEmbedder(),
        store=FakeStore(),
        session_factory=_factory,
        stages=StageConfig(
            routing_enabled=False,
            query_plan_enabled=False,
            search_mode="vector",
            rerank=False,
            top_k=5,
            distance_threshold=2.0,
        ),
        reference_date=date(2025, 3, 1),
        explicit_collections=[Collection.TECHNICAL_DOC],
    )

    assert outcome.chunks == []
    assert outcome.low_confidence is True
