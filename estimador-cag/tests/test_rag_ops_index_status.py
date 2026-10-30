"""Tests de los avisos operativos de ``index_status``."""

from __future__ import annotations

from scripts.rag_ops.index_status import _print_warnings


def test_warns_when_there_is_no_vector_index(capsys) -> None:
    _print_warnings({"indexes": [{"index_type": "btree", "scans": 10}]})

    output = capsys.readouterr().out
    assert "no hay índice vectorial" in output


def test_warns_when_vector_index_has_no_scans(capsys) -> None:
    _print_warnings(
        {"indexes": [{"index_name": "chunks_embedding_idx", "index_type": "hnsw", "scans": 0}]}
    )

    output = capsys.readouterr().out
    assert "chunks_embedding_idx" in output
    assert "sin ningún scan" in output


def test_does_not_warn_when_vector_index_is_used(capsys) -> None:
    _print_warnings(
        {"indexes": [{"index_name": "chunks_embedding_idx", "index_type": "hnsw", "scans": 12}]}
    )

    assert capsys.readouterr().out == ""
