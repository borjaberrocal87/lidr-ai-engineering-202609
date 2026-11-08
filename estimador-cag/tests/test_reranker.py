"""Unit tests for the cross-encoder reranker wrapper.

A FAKE scorer is injected (``reranker._model``) so the tests never download or
load torch weights — they exercise the wrapper's ordering/truncation logic, not
the model.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.generation.rag.retrieval.reranker import CrossEncoderReranker


@dataclass
class _Candidate:
    id: int
    content: str


class _FakeModel:
    """Stands in for sentence_transformers.CrossEncoder.

    ``predict`` returns a score per (query, document) pair driven by a mapping
    from document text → score, so tests control the ranking deterministically.
    """

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self.scores_by_text = scores_by_text
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        return [self.scores_by_text[document] for _query, document in pairs]


def _reranker_with(scores_by_text: dict[str, float]) -> tuple[CrossEncoderReranker, _FakeModel]:
    reranker = CrossEncoderReranker("fake-model")
    fake = _FakeModel(scores_by_text)
    reranker._model = fake  # bypass lazy load
    return reranker, fake


def test_rerank_orders_by_descending_score() -> None:
    candidates = [
        _Candidate(1, "weakly relevant"),
        _Candidate(2, "highly relevant"),
        _Candidate(3, "not relevant"),
    ]
    reranker, _ = _reranker_with(
        {"weakly relevant": 0.4, "highly relevant": 0.9, "not relevant": 0.1}
    )

    ordered = reranker.rerank("q", candidates, top_n=3)

    assert [candidate.id for candidate in ordered] == [2, 1, 3]


def test_rerank_truncates_to_top_n() -> None:
    candidates = [_Candidate(index, f"doc{index}") for index in range(5)]
    reranker, _ = _reranker_with({f"doc{index}": float(index) for index in range(5)})

    ordered = reranker.rerank("q", candidates, top_n=2)

    assert [candidate.id for candidate in ordered] == [4, 3]


def test_rerank_empty_candidates_returns_empty_without_scoring() -> None:
    reranker, fake = _reranker_with({})

    assert reranker.rerank("q", [], top_n=5) == []
    assert fake.calls == []


def test_score_builds_query_document_pairs_in_order() -> None:
    reranker, fake = _reranker_with({"a": 0.2, "b": 0.5})

    scores = reranker.score("my query", ["a", "b"])

    assert scores == [0.2, 0.5]
    assert fake.calls == [[("my query", "a"), ("my query", "b")]]


def test_rerank_accepts_custom_text_extractor() -> None:
    candidates = [{"id": 1, "body": "low"}, {"id": 2, "body": "high"}]
    reranker, _ = _reranker_with({"low": 0.1, "high": 0.8})

    ordered = reranker.rerank(
        "q",
        candidates,
        top_n=2,
        text_of=lambda candidate: candidate["body"],
    )

    assert [candidate["id"] for candidate in ordered] == [2, 1]
