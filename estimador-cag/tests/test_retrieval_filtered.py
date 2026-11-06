"""Retrieval S09: filtros estructurales, umbral y soft-fail."""

from __future__ import annotations

from typing import Any

from app.generation.rag.retriever import SemanticRetriever
from app.generation.rag.store.repository import ChunkMatch


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_one(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2, 0.3]


class _SessionContext:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *args: object) -> bool:
        return False


class FakeSessionFactory:
    def __call__(self) -> _SessionContext:
        return _SessionContext()


class FakeStore:
    def __init__(self, matches: list[ChunkMatch], candidates: int = 0) -> None:
        self.matches = matches
        self.candidates = candidates
        self.calls: list[dict[str, Any]] = []
        self.search_mode = "vector"

    async def search_with_filters(self, session: object, **kwargs: Any):
        self.calls.append(kwargs)
        return self.matches, self.candidates


def _match(
    *,
    chunk_id: int = 7,
    metadata: dict[str, Any] | None = None,
    distance: float = 0.25,
) -> ChunkMatch:
    return ChunkMatch(
        chunk_id=chunk_id,
        document_id=3,
        chunk_type="historical_task",
        content="[Module: Auth] OAuth login",
        distance=distance,
        metadata=metadata if metadata is not None else {"client_sector": "finance", "year": 2024},
    )


def _retriever(store: FakeStore) -> SemanticRetriever:
    return SemanticRetriever(
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        session_factory=FakeSessionFactory(),  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
    )


async def test_retrieve_chunks_flattens_metadata_into_typed_shape() -> None:
    store = FakeStore([_match()], candidates=12)

    outcome = await _retriever(store).retrieve_chunks(
        query_text="historico de pagos",
        top_k=5,
        distance_threshold=0.6,
        sectors=["finance"],
    )

    assert outcome.low_confidence is False
    assert outcome.candidates_evaluated == 12
    chunk = outcome.chunks[0]
    assert (chunk.id, chunk.sector, chunk.project_year) == (7, "finance", 2024)
    assert store.calls[0]["sectors"] == ["finance"]
    assert store.calls[0]["top_k"] == 5


async def test_retrieve_chunks_returns_low_confidence_when_no_matches() -> None:
    store = FakeStore([], candidates=4)

    outcome = await _retriever(store).retrieve_chunks(
        query_text="proyecto sin analogos",
        top_k=10,
        distance_threshold=0.2,
    )

    assert outcome.chunks == []
    assert outcome.low_confidence is True


async def test_retrieve_chunks_tolerates_missing_year_and_sector() -> None:
    store = FakeStore([_match(metadata={})], candidates=1)

    outcome = await _retriever(store).retrieve_chunks(query_text="proyecto", top_k=3)

    assert outcome.chunks[0].project_year is None
    assert outcome.chunks[0].sector is None
