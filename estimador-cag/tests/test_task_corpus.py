"""Corpus task-level: determinismo, validación y chunking con módulo."""

from __future__ import annotations

from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.schemas import Budget
from scripts.build_task_corpus import CHUNK_TYPE, DOCUMENT_TYPE, generate_corpus


def test_task_corpus_generation_is_deterministic() -> None:
    first = generate_corpus(count=3, seed=90)
    second = generate_corpus(count=3, seed=90)

    assert first == second


def test_every_task_is_a_valid_component_with_a_module() -> None:
    corpus = generate_corpus(count=2, seed=90)

    for project in corpus:
        budget = Budget.model_validate(project)
        assert budget.components
        for component in budget.components:
            assert component.module


def test_total_hours_match_the_sum_of_tasks() -> None:
    corpus = generate_corpus(count=2, seed=90)

    for project in corpus:
        assert project["total_estimated_hours"] == sum(
            component["estimated_hours"] for component in project["components"]
        )


def test_structural_chunker_emits_module_in_text_and_metadata() -> None:
    corpus = generate_corpus(count=1, seed=90)
    budget = Budget.model_validate(corpus[0])

    chunk = JSONStructuralChunker().chunk([budget])[0]

    assert "Module:" in chunk.text
    assert chunk.metadata["module"]


def test_corpus_constants_are_stable() -> None:
    assert DOCUMENT_TYPE == "historical_task_breakdown"
    assert CHUNK_TYPE == "historical_task"
