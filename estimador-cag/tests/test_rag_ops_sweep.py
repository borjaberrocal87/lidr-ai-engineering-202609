"""Tests de la lógica de decisión del barrido de ef_search."""

from __future__ import annotations

from scripts.rag_ops.ef_search_sweep import _recall_at_k, _recommend_ef


def test_recall_at_k_counts_top_k_overlap() -> None:
    assert _recall_at_k([1, 2, 3, 4, 5], [2, 1, 9, 8, 7], 5) == 0.4
    assert _recall_at_k([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 5) == 1.0
    assert _recall_at_k([1, 2], [3, 4], 2) == 0.0


def test_recommendation_prefers_smallest_ef_within_tolerance() -> None:
    results = [
        (10, 0.90, 10.0, 12.0),
        (20, 0.98, 12.0, 15.0),
        (40, 0.99, 15.0, 18.0),
        (80, 1.00, 20.0, 25.0),
    ]

    assert _recommend_ef(results) == 40


def test_recommendation_falls_back_to_last_value() -> None:
    results = [
        (10, 0.50, 10.0, 12.0),
        (20, 0.60, 12.0, 15.0),
    ]

    assert _recommend_ef(results) == 20
