"""Tests for the lexical branch SQL (no live Postgres)."""

from __future__ import annotations

from sqlalchemy.dialects import postgresql

from app.generation.rag.store.repository import ChunkStore, _or_tsquery_text


def _compiled_lexical(store: ChunkStore | None = None) -> str:
    statement = (store or ChunkStore())._lexical_statement(
        query_text="ecommerce checkout",
        top_k=50,
    )
    return str(statement.compile(dialect=postgresql.dialect()))


def test_lexical_statement_uses_full_text_operator_and_rank() -> None:
    compiled = _compiled_lexical().lower()

    assert "content_tsv" in compiled
    assert "@@" in compiled
    assert "ts_rank_cd" in compiled
    assert "websearch_to_tsquery" in compiled


def test_or_tsquery_text_joins_terms_with_or() -> None:
    assert _or_tsquery_text("headless ecommerce storefront") == (
        "headless or ecommerce or storefront"
    )


def test_lexical_statement_orders_by_rank_desc_and_limits() -> None:
    compiled = _compiled_lexical().upper()

    assert "ORDER BY" in compiled
    assert "DESC" in compiled
    assert "LIMIT" in compiled


def test_lexical_statement_applies_structural_filters() -> None:
    statement = ChunkStore()._lexical_statement(
        query_text="ecommerce checkout",
        top_k=10,
        sectors=["ecommerce"],
        project_year_min=2020,
        chunk_types=["budget_component"],
    )
    compiled = str(statement.compile(dialect=postgresql.dialect())).lower()

    assert "chunks.metadata ->>" in compiled
    assert "chunks.chunk_type in" in compiled
    assert ">=" in compiled
