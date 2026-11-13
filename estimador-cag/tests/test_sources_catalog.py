"""Contrato del catálogo de fuentes (``data/catalog/collections.yaml``)."""

from __future__ import annotations

from datetime import date

from app.generation.rag.retrieval.sources import (
    Collection,
    HardFilters,
    collections,
    match_rules,
    spec_for,
)
from app.generation.rag.store.repository import ChunkMatch


def test_catalog_loads_the_three_sources() -> None:
    catalog = collections()

    assert set(catalog) == {
        Collection.BUDGET,
        Collection.TRANSCRIPT,
        Collection.TECHNICAL_DOC,
    }
    assert catalog[Collection.BUDGET].label == "Presupuestos"
    assert catalog[Collection.TRANSCRIPT].label == "Transcripciones"
    assert catalog[Collection.TECHNICAL_DOC].label == "Documentación técnica"


def test_rules_match_spanish_and_english_vocabulary() -> None:
    assert match_rules("presupuesto y coste de horas") == [Collection.BUDGET]
    assert match_rules("reunión con el cliente") == [Collection.TRANSCRIPT]
    assert match_rules("runbook de arquitectura") == [Collection.TECHNICAL_DOC]
    assert match_rules("consulta completamente neutra") == []


def test_to_chunk_preserves_provenance_and_dates() -> None:
    spec = spec_for(Collection.TRANSCRIPT)
    match = ChunkMatch(
        chunk_id=7,
        document_id=3,
        chunk_type="meeting_segment",
        content="The client confirmed the OAuth scope.",
        distance=0.2,
        metadata={
            "transcript_id": "TR-2025-101",
            "meeting_date": "2025-02-18",
        },
    )

    chunk = spec.to_chunk(match, distance=0.2)

    assert chunk.id == 7
    assert chunk.document_id == 3
    assert chunk.collection == "transcript"
    assert chunk.source_id == "TR-2025-101"
    assert chunk.document_date == date(2025, 2, 18)
    assert chunk.sector is None
    assert chunk.project_year is None


def test_budget_chunk_flattens_hours_and_sector() -> None:
    spec = spec_for(Collection.BUDGET)
    match = ChunkMatch(
        chunk_id=1,
        document_id=1,
        chunk_type="historical_task",
        content="OAuth2 login",
        distance=0.1,
        metadata={
            "budget_id": "TASK-2022-0001",
            "client_sector": "finance",
            "year": 2022,
            "estimated_hours": 40,
        },
    )

    chunk = spec.to_chunk(match, distance=0.1)

    assert chunk.collection == "budget"
    assert chunk.source_id == "TASK-2022-0001"
    assert chunk.sector == "finance"
    assert chunk.project_year == 2022
    assert chunk.estimated_hours == 40
    assert chunk.document_date == date(2022, 1, 1)


def test_hard_filters_only_emit_axes_the_source_understands() -> None:
    filters = HardFilters(
        technologies=("python",),
        sectors=("finance",),
        min_date=date(2024, 1, 1),
        version="v2.3",
    )

    assert len(spec_for(Collection.BUDGET).hard_filter_clauses(filters)) == 3
    assert len(spec_for(Collection.TRANSCRIPT).hard_filter_clauses(filters)) == 1
    assert len(spec_for(Collection.TECHNICAL_DOC).hard_filter_clauses(filters)) == 1
