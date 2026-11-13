"""Vigencia temporal: peso exponencial y reordenación de finalistas."""

from __future__ import annotations

from datetime import date

from app.generation.rag.retrieval.freshness import apply_freshness, freshness_weight
from app.generation.rag.schemas import RetrievedChunk


def _chunk(chunk_id: int, *, document_date: date | None) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id=chunk_id,
        content=f"chunk {chunk_id}",
        chunk_type="budget_component",
        distance=0.1,
        document_date=document_date,
    )


def test_weight_is_one_for_fresh_or_undated_chunks() -> None:
    assert freshness_weight(0, 900) == 1.0
    assert freshness_weight(-5, 900) == 1.0
    assert freshness_weight(100, 0) == 1.0


def test_weight_halves_at_one_half_life() -> None:
    assert freshness_weight(900, 900) == 0.5


def test_decay_can_change_the_order_of_finalists() -> None:
    reference = date(2025, 1, 1)
    old = _chunk(1, document_date=date(2015, 1, 1))
    recent = _chunk(2, document_date=date(2024, 12, 1))

    reordered = apply_freshness(
        [(old, 1.0), (recent, 0.8)],
        half_life_days=900,
        reference_date=reference,
    )

    assert [chunk.id for chunk, _score in reordered] == [2, 1]


def test_chunks_without_date_keep_their_base_score() -> None:
    reference = date(2025, 1, 1)
    undated = _chunk(1, document_date=None)

    reordered = apply_freshness([(undated, 0.7)], half_life_days=900, reference_date=reference)

    assert reordered == [(undated, 0.7)]


def test_empty_input_is_empty() -> None:
    assert apply_freshness([], half_life_days=900, reference_date=date(2025, 1, 1)) == []
