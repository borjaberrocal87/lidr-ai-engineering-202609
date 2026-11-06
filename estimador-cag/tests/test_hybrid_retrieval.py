"""Unit tests for the Session 10 recall-then-rerank pipeline.

The vector store, lexical branch and cross-encoder are all faked: no Postgres,
no torch. We assert the four configurations compose correctly — recall width,
RRF fusion of the two branches, reranker reordering + top-N cut, and the
soft-fail contract.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.generation.rag.retrieval.pipeline import retrieve
from app.generation.rag.store.repository import ChunkMatch


class FakeSession:
    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _factory() -> FakeSession:
    return FakeSession()


def _match(
    chunk_id: int,
    *,
    distance: float = 0.3,
    budget_id: str | None = None,
) -> ChunkMatch:
    return ChunkMatch(
        chunk_id=chunk_id,
        document_id=chunk_id * 10,
        chunk_type="budget_component",
        content=f"chunk-{chunk_id}",
        distance=distance,
        metadata={
            "client_sector": "ecommerce",
            "year": 2024,
            "budget_id": budget_id or f"BUD-{chunk_id}",
        },
    )


class FakeStore:
    search_mode = "vector"

    def __init__(
        self,
        *,
        vector_rows: list[ChunkMatch],
        lexical_rows: list[ChunkMatch],
        candidates: int = 12,
    ) -> None:
        self.vector_rows = vector_rows
        self.lexical_rows = lexical_rows
        self.candidates = candidates
        self.vector_calls: list[dict[str, Any]] = []
        self.lexical_calls: list[dict[str, Any]] = []

    async def search_with_filters(
        self,
        session: object,
        **kwargs: Any,
    ) -> tuple[list[ChunkMatch], int]:
        self.vector_calls.append(kwargs)
        return self.vector_rows[: kwargs["top_k"]], self.candidates

    async def search_lexical(
        self,
        session: object,
        **kwargs: Any,
    ) -> list[ChunkMatch]:
        self.lexical_calls.append(kwargs)
        return self.lexical_rows[: kwargs["top_k"]]


class FakeReranker:
    """Reorders candidates by a content→score map and cuts to top_n."""

    def __init__(self, scores_by_content: dict[str, float]) -> None:
        self.scores_by_content = scores_by_content

    def rerank(
        self,
        query: str,
        candidates: list[Any],
        *,
        top_n: int,
        text_of: Any = lambda candidate: candidate.content,
    ) -> list[Any]:
        ranked = sorted(
            candidates,
            key=lambda candidate: self.scores_by_content[text_of(candidate)],
            reverse=True,
        )
        return ranked[:top_n]


async def test_vector_mode_no_rerank_returns_distance_order() -> None:
    store = FakeStore(
        vector_rows=[_match(1), _match(2), _match(3)],
        lexical_rows=[],
    )

    result = await retrieve(
        store=store,  # type: ignore[arg-type]
        session_factory=_factory,  # type: ignore[arg-type]
        query_embedding=[0.0] * 1536,
        query_text="ecommerce checkout",
        search_mode="vector",
        rerank=False,
        top_k=2,
    )

    assert [chunk.id for chunk in result.chunks] == [1, 2]
    assert result.low_confidence is False
    assert result.candidates_evaluated == 12
    assert result.chunks[0].budget_id == "BUD-1"
    assert store.vector_calls[0]["top_k"] == 2
    assert store.lexical_calls == []


async def test_hybrid_mode_fuses_vector_and_lexical() -> None:
    store = FakeStore(
        vector_rows=[_match(1), _match(2)],
        lexical_rows=[_match(3), _match(1)],
    )

    result = await retrieve(
        store=store,  # type: ignore[arg-type]
        session_factory=_factory,  # type: ignore[arg-type]
        query_embedding=[0.0] * 1536,
        query_text="ecommerce checkout",
        search_mode="hybrid",
        rerank=False,
        top_k=5,
        recall_k=50,
    )

    assert {chunk.id for chunk in result.chunks} == {1, 2, 3}
    assert store.vector_calls[0]["top_k"] == 50
    assert store.lexical_calls[0]["top_k"] == 50


async def test_rerank_reorders_and_cuts_to_top_n() -> None:
    store = FakeStore(
        vector_rows=[_match(1), _match(2), _match(3)],
        lexical_rows=[],
    )
    reranker = FakeReranker({"chunk-1": 0.1, "chunk-2": 0.5, "chunk-3": 0.9})

    result = await retrieve(
        store=store,  # type: ignore[arg-type]
        session_factory=_factory,  # type: ignore[arg-type]
        query_embedding=[0.0] * 1536,
        query_text="ecommerce checkout",
        search_mode="vector",
        rerank=True,
        recall_k=50,
        rerank_top_n=2,
        reranker=reranker,  # type: ignore[arg-type]
    )

    assert [chunk.id for chunk in result.chunks] == [3, 2]
    assert store.vector_calls[0]["top_k"] == 50


async def test_soft_fail_when_nothing_retrieved() -> None:
    store = FakeStore(vector_rows=[], lexical_rows=[], candidates=7)

    result = await retrieve(
        store=store,  # type: ignore[arg-type]
        session_factory=_factory,  # type: ignore[arg-type]
        query_embedding=[0.0] * 1536,
        query_text="ecommerce checkout",
        search_mode="hybrid",
        rerank=False,
    )

    assert result.chunks == []
    assert result.low_confidence is True
    assert result.candidates_evaluated == 7


async def test_invalid_search_mode_is_rejected() -> None:
    store = FakeStore(vector_rows=[], lexical_rows=[])

    with pytest.raises(ValueError, match="search_mode"):
        await retrieve(
            store=store,  # type: ignore[arg-type]
            session_factory=_factory,  # type: ignore[arg-type]
            query_embedding=[0.0] * 1536,
            query_text="q",
            search_mode="magic",
        )


async def test_rerank_requires_a_reranker() -> None:
    store = FakeStore(vector_rows=[_match(1)], lexical_rows=[])

    with pytest.raises(ValueError, match="reranker"):
        await retrieve(
            store=store,  # type: ignore[arg-type]
            session_factory=_factory,  # type: ignore[arg-type]
            query_embedding=[0.0] * 1536,
            query_text="q",
            rerank=True,
        )
