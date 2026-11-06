"""Aumentación: bloques <source> y presupuesto de tokens."""

from __future__ import annotations

from app.generation.rag.context_builder import (
    assemble_context,
    fit_chunks_to_budget,
    render_source,
)
from app.generation.rag.schemas import RetrievedChunk


class CharEncoder:
    """Encoder de juguete: un token por carácter."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


def _chunk(chunk_id: int, content: str, distance: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        document_id=1,
        content=content,
        sector="finance",
        project_year=2024,
        chunk_type="budget_component",
        distance=distance,
    )


def test_assemble_context_wraps_each_chunk_with_its_metadata() -> None:
    block = assemble_context([_chunk(7, "OAuth backend", 0.1234)])

    assert '<source id="7" sector="finance" project_year="2024"' in block
    assert 'chunk_type="budget_component" distance="0.1234"' in block
    assert "OAuth backend" in block
    assert block.endswith("</source>")


def test_fit_chunks_to_budget_keeps_whole_chunks_in_order() -> None:
    first = _chunk(1, "primero", 0.1)
    second = _chunk(2, "segundo", 0.2)
    encoder = CharEncoder()
    budget = len(encoder.encode(render_source(first))) + 1

    kept = fit_chunks_to_budget([first, second], budget, encoder)

    assert kept == [first]


def test_fit_chunks_to_budget_never_splits_a_chunk() -> None:
    chunk = _chunk(1, "x" * 500, 0.1)
    encoder = CharEncoder()
    budget = len(encoder.encode(render_source(chunk))) - 1

    kept = fit_chunks_to_budget([chunk], budget, encoder)

    assert kept == []


def test_empty_context_is_an_empty_string() -> None:
    assert assemble_context([]) == ""
