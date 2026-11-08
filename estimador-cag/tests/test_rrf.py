"""Unit tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from app.generation.rag.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_rewards_documents_present_in_both_rankings() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)

    ids = [chunk_id for chunk_id, _score in fused]
    assert ids[0] == 1  # 1st in one branch, 2nd in the other
    assert set(ids) == {1, 2, 3, 4}


def test_rrf_ignores_duplicate_ids_within_a_ranking() -> None:
    fused = reciprocal_rank_fusion([[7, 7, 8]], k=60)

    ids = [chunk_id for chunk_id, _score in fused]
    assert ids == [7, 8]


def test_rrf_ties_break_by_ascending_id() -> None:
    fused = reciprocal_rank_fusion([[2], [1]], k=60)

    assert [chunk_id for chunk_id, _score in fused] == [1, 2]


def test_rrf_empty_rankings_return_empty() -> None:
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([[1]], k=0)
