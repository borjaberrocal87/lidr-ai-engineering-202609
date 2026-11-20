"""Augmentación del contexto: compresión extractiva y reordenado por extremos."""

from __future__ import annotations

from app.generation.rag.quality.augmentation import (
    AugmentationPolicy,
    apply_augmentation,
    compress_for_context,
    key_lines,
    reorder_strong_edges,
)
from app.generation.rag.schemas import RetrievedChunk


def _chunk(chunk_id: int, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id=1,
        content=content,
        chunk_type="budget_component",
        distance=0.1 * chunk_id,
    )


def test_key_lines_keeps_figures_and_ids_and_drops_prose() -> None:
    content = (
        "Component: Checkout\n"
        "El equipo dedicó varias sesiones a alinear el alcance con negocio.\n"
        "Estimated hours: 48\n"
        "ref::BUDGET-2024-0001\n"
    )
    assert key_lines(content) == [
        "Component: Checkout",
        "Estimated hours: 48",
        "ref::BUDGET-2024-0001",
    ]


def test_key_lines_falls_back_to_first_line_when_no_anchor() -> None:
    assert key_lines("Una nota sin cifras\ncon más prosa") == ["Una nota sin cifras"]


def test_compress_preserves_chunk_identity() -> None:
    chunk = _chunk(7, "Component: Auth\nprosa irrelevante\nEstimated hours: 16")
    compressed = compress_for_context(chunk)

    assert compressed.id == 7
    assert compressed.document_id == chunk.document_id
    assert compressed.distance == chunk.distance
    assert compressed.content == "Component: Auth\nEstimated hours: 16"


def test_reorder_puts_second_best_at_the_tail() -> None:
    chunks = [_chunk(i, f"c{i}") for i in range(5)]

    reordered = reorder_strong_edges(chunks)

    assert [chunk.id for chunk in reordered] == [0, 2, 4, 3, 1]


def test_reorder_keeps_tiny_lists_stable() -> None:
    chunks = [_chunk(0, "a"), _chunk(1, "b")]

    assert [chunk.id for chunk in reorder_strong_edges(chunks)] == [0, 1]


def test_apply_augmentation_can_disable_each_pass() -> None:
    chunks = [_chunk(0, "Component: A\nEstimated hours: 8"), _chunk(1, "prosa")]

    untouched = apply_augmentation(chunks, AugmentationPolicy(compress=False, reorder=False))
    assert [chunk.content for chunk in untouched] == [chunk.content for chunk in chunks]

    compressed = apply_augmentation(chunks, AugmentationPolicy(compress=True, reorder=False))
    assert compressed[1].content == "prosa"
