"""Tests de las utilidades compartidas de RAG Ops."""

from __future__ import annotations

from scripts.rag_ops.common import (
    format_table,
    latency_stats,
    latency_summary,
    truncate,
    vector_literal,
)


def test_format_table_aligns_headers_and_rows() -> None:
    table = format_table(["nombre", "ms"], [["a", "1.0"], ["largo", "22.0"]])

    lines = table.splitlines()
    assert lines[0].split() == ["nombre", "ms"]
    assert lines[1] == "------  ----"
    assert lines[2].split() == ["a", "1.0"]
    assert lines[3].split() == ["largo", "22.0"]


def test_truncate_respects_width_and_adds_ellipsis() -> None:
    assert truncate("corto", 10) == "corto"
    assert truncate("consulta muy larga", 8) == "consult…"


def test_latency_stats_returns_mean_and_population_deviation() -> None:
    mean, deviation = latency_stats([10.0, 20.0, 30.0])

    assert mean == 20.0
    assert round(deviation, 4) == 8.165


def test_latency_summary_includes_p95_only_with_enough_samples() -> None:
    assert "p95" not in latency_summary([1.0, 2.0, 3.0])
    assert "p95" in latency_summary([float(value) for value in range(25)])


def test_vector_literal_uses_pgvector_format() -> None:
    assert vector_literal([0.5, -1.25]) == "[0.5,-1.25]"
