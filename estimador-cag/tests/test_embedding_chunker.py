"""Tests del chunker estructural y de los contratos del pipeline de embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.embedding_pipeline.chunker import JSONStructuralChunker, count_tokens
from app.embedding_pipeline.schemas import Budget, IngestRequest

DATASET = Path(__file__).resolve().parents[1] / "data" / "budgets_sample.json"


def _budgets() -> list[Budget]:
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    return IngestRequest(budgets=payload).budgets


def test_one_chunk_per_component() -> None:
    budgets = _budgets()
    chunks = JSONStructuralChunker().chunk(budgets)

    assert len(budgets) == 17
    assert sum(len(budget.components) for budget in budgets) == 60
    assert len(chunks) == 60


def test_chunk_is_traceable_and_carries_filterable_metadata() -> None:
    first = JSONStructuralChunker().chunk(_budgets())[0]

    assert first.chunk_id == "BUD-2024-001::AUTH-001"
    assert first.metadata == {
        "budget_id": "BUD-2024-001",
        "component_id": "AUTH-001",
        "client_sector": "finance",
        "main_technology": "ruby_on_rails",
        "year": 2024,
        "complexity": "high",
        "estimated_hours": 120,
    }


def test_text_prepends_the_parent_context_header() -> None:
    first = JSONStructuralChunker().chunk(_budgets())[0]

    assert first.text.startswith("[Project: Mobile banking API")
    assert "[Client sector: finance | Year: 2024 | Main tech: ruby_on_rails]" in first.text
    assert "Component: OAuth 2.0 authentication backend" in first.text


def test_token_count_uses_the_embedding_tokenizer() -> None:
    first = JSONStructuralChunker().chunk(_budgets())[0]

    assert first.token_count == count_tokens(first.text)
    assert first.token_count > 0


def test_unknown_sector_fails_validation() -> None:
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    payload[0]["client_metadata"]["sector"] = "retail"

    with pytest.raises(ValidationError):
        IngestRequest(budgets=payload)


def test_unknown_field_fails_validation() -> None:
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    payload[0]["surprise_field"] = "nope"

    with pytest.raises(ValidationError):
        IngestRequest(budgets=payload)
