"""Tests del generador de corpus sintético."""

from __future__ import annotations

from scripts.rag_ops.corpus_seed import _build_chunks


def test_build_chunks_is_deterministic_for_a_seed() -> None:
    first = _build_chunks(5, seed=8)
    second = _build_chunks(5, seed=8)

    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]


def test_build_chunks_changes_with_a_different_seed() -> None:
    first = _build_chunks(5, seed=8)
    second = _build_chunks(5, seed=9)

    assert [chunk.text for chunk in first] != [chunk.text for chunk in second]


def test_build_chunks_assigns_stable_ids_metadata_and_tokens() -> None:
    chunks = _build_chunks(3, seed=8)

    assert [chunk.chunk_id for chunk in chunks] == [
        "synthetic::000000",
        "synthetic::000001",
        "synthetic::000002",
    ]
    assert [chunk.metadata["index"] for chunk in chunks] == [0, 1, 2]
    assert all(chunk.metadata["synthetic"] == "true" for chunk in chunks)
    assert all(chunk.token_count > 0 for chunk in chunks)
