"""Tests del lector de planes de ``index_compare``."""

from __future__ import annotations

from scripts.rag_ops.index_compare import _find_index_name


def test_finds_index_in_leaf_node() -> None:
    plan = {"Node Type": "Index Scan", "Index Name": "chunks_embedding_idx"}

    assert _find_index_name(plan) == "chunks_embedding_idx"


def test_finds_index_nested_under_sort_or_limit() -> None:
    plan = {
        "Node Type": "Limit",
        "Plans": [
            {
                "Node Type": "Sort",
                "Plans": [
                    {"Node Type": "Index Scan", "Index Name": "chunks_embedding_halfvec_idx"}
                ],
            }
        ],
    }

    assert _find_index_name(plan) == "chunks_embedding_halfvec_idx"


def test_returns_dash_when_plan_is_sequential() -> None:
    plan = {"Node Type": "Seq Scan", "Plans": [{"Node Type": "Sort"}]}

    assert _find_index_name(plan) == "—"
