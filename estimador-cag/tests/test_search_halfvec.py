"""Tests del modo de búsqueda vectorial vs halfvec del ``ChunkStore``."""

from __future__ import annotations

from sqlalchemy.dialects import postgresql

from app.embedding_pipeline.store.repository import ChunkStore


def _compiled(store: ChunkStore) -> str:
    statement = store._search_statement([0.1] * 1536, 5)
    return str(statement.compile(dialect=postgresql.dialect()))


def test_search_mode_defaults_to_full_vector() -> None:
    store = ChunkStore()

    compiled = _compiled(store)

    assert store.search_mode == "vector"
    assert "halfvec" not in compiled.lower()
    assert "<=>" in compiled
    assert "chunks.embedding IS NOT NULL" in compiled


def test_search_mode_uses_halfvec_when_enabled() -> None:
    store = ChunkStore(use_halfvec=True)

    compiled = _compiled(store)

    assert store.search_mode == "halfvec"
    assert "halfvec(1536)" in compiled.lower()
    assert "<=>" in compiled


def test_search_statement_orders_by_distance_and_limits() -> None:
    compiled = _compiled(ChunkStore())

    assert "ORDER BY" in compiled
    assert "LIMIT" in compiled
