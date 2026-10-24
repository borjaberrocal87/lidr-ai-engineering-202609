"""Tests de las estrategias de chunking de la Sesión 7 (directo)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.embedding_pipeline.analysis.comparison import ChunkingComparator
from app.embedding_pipeline.schemas import Budget, Chunk, EmbeddedChunk
from app.embedding_pipeline.strategies import (
    ContextualRetrievalChunker,
    FixedSizeChunker,
    HierarchicalChunker,
    PropositionalChunker,
    RecursiveChunker,
    SentenceWindowChunker,
)
from app.embedding_pipeline.strategies import sentence_window as sentence_window_module
from app.embedding_pipeline.structural import JSONStructuralChunker

DATASET = Path(__file__).resolve().parents[1] / "data" / "budgets_sample.json"


def _budgets() -> list[Budget]:
    return [Budget(**item) for item in json.loads(DATASET.read_text(encoding="utf-8"))]


@pytest.mark.parametrize(
    "chunker",
    [
        JSONStructuralChunker(),
        FixedSizeChunker(),
        RecursiveChunker(),
        HierarchicalChunker(),
    ],
)
def test_offline_strategies_produce_chunks(chunker: Any) -> None:
    chunks = chunker.chunk(_budgets())

    assert len(chunks) > 0
    assert all(chunk.token_count >= 0 for chunk in chunks)
    assert all(chunk.chunk_id for chunk in chunks)
    assert chunker.last_extra_api_calls == 0
    assert chunker.last_extra_cost_usd == 0.0


def test_sentence_window_uses_window_metadata(monkeypatch) -> None:
    monkeypatch.setattr(sentence_window_module, "_ensure_punkt", lambda: None)
    monkeypatch.setattr(
        sentence_window_module.nltk,
        "sent_tokenize",
        lambda text: [part.strip() for part in text.split(".") if part.strip()],
    )

    chunks = SentenceWindowChunker().chunk(_budgets())

    assert len(chunks) > 0
    assert all("window_text" in chunk.metadata for chunk in chunks)


class _FakeCompletions:
    def create(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"propositions": ["hecho uno", "hecho dos"]}')
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
        )


def test_propositional_chunker_with_fake_llm() -> None:
    client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))
    chunker = PropositionalChunker(client=client, model="gpt-4o-mini")  # type: ignore[arg-type]

    chunks = chunker.chunk(_budgets()[:1])

    assert len(chunks) == 8
    assert chunker.last_extra_api_calls == 4
    assert chunker.last_extra_cost_usd > 0


class _FakeAnthropicMessages:
    def create(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            content=[SimpleNamespace(text="contexto del presupuesto")],
            usage=SimpleNamespace(
                input_tokens=20,
                output_tokens=5,
                cache_creation_input_tokens=10,
                cache_read_input_tokens=0,
            ),
        )


def test_contextual_retrieval_chunker_with_fake_llm() -> None:
    client = SimpleNamespace(messages=_FakeAnthropicMessages())
    chunker = ContextualRetrievalChunker(client=client, model="claude-sonnet-4-5")  # type: ignore[arg-type]

    chunks = chunker.chunk(_budgets()[:1])

    assert len(chunks) == 4
    assert chunks[0].text.startswith("contexto del presupuesto")
    assert chunker.last_extra_api_calls == 4
    assert chunker.last_extra_cost_usd > 0


class _FakeEmbedder:
    def embed_one(self, text: str) -> list[float]:
        return [1.0, float(len(text) % 5)]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return [
            EmbeddedChunk(**chunk.model_dump(), embedding=self.embed_one(chunk.text))
            for chunk in chunks
        ]


def test_comparator_computes_stats_and_top_k() -> None:
    budgets = _budgets()
    comparator = ChunkingComparator(
        [JSONStructuralChunker(), HierarchicalChunker()], _FakeEmbedder()
    )

    stats = comparator.compute_stats(budgets)
    queries = comparator.run_queries(budgets, ["OAuth authentication"], top_k=2)

    assert set(stats) == {"structural", "hierarchical"}
    assert stats["structural"].n_chunks == 60
    assert stats["structural"].n_orphan_chunks >= 0
    assert stats["hierarchical"].n_chunks > 60
    assert len(queries["structural"]) == 1
    assert len(queries["structural"][0].top_k) == 2
    assert queries["structural"][0].top_k[0].cosine >= queries["structural"][0].top_k[1].cosine
