"""Tests de los modelos ORM del store vectorial (multi-índice S10)."""

from __future__ import annotations

from app.foundation.persistence.models import Base
from app.generation.rag.store.models import (
    EMBEDDING_DIMENSIONS,
    BudgetChunkRow,
    ChunkRow,
    DocumentRow,
    TechnicalDocChunkRow,
    TranscriptChunkRow,
)


def test_store_tables_are_registered_in_the_shared_base() -> None:
    assert DocumentRow.__tablename__ == "documents"
    assert BudgetChunkRow.__tablename__ == "budget_chunks"
    assert TranscriptChunkRow.__tablename__ == "transcript_chunks"
    assert TechnicalDocChunkRow.__tablename__ == "technical_doc_chunks"
    assert ChunkRow is BudgetChunkRow, "El alias S08/S09 apunta a presupuestos"
    assert DocumentRow.metadata.tables["documents"] is DocumentRow.__table__
    assert DocumentRow.metadata.tables["budget_chunks"] is BudgetChunkRow.__table__
    assert DocumentRow.metadata.tables["transcript_chunks"] is TranscriptChunkRow.__table__
    assert (
        DocumentRow.metadata.tables["technical_doc_chunks"]
        is TechnicalDocChunkRow.__table__
    )
    assert Base.metadata is DocumentRow.metadata


def test_chunks_belong_to_documents_with_cascade() -> None:
    foreign_keys = ChunkRow.__table__.c.document_id.foreign_keys

    assert len(foreign_keys) == 1
    assert next(iter(foreign_keys)).ondelete == "CASCADE"


def test_embedding_dimension_is_fixed() -> None:
    assert EMBEDDING_DIMENSIONS == 1536
    assert ChunkRow.__table__.c.embedding.nullable is True


def test_chunks_have_relational_indexes() -> None:
    indexed_columns = {column for index in ChunkRow.__table__.indexes for column in index.columns}

    assert {column.name for column in indexed_columns} == {
        "chunk_type",
        "document_id",
        "metadata",
        "content_tsv",
    }


def test_every_collection_has_its_own_indexes() -> None:
    for model in (BudgetChunkRow, TranscriptChunkRow, TechnicalDocChunkRow):
        names = {index.name for index in model.__table__.indexes}
        assert f"ix_{model.__tablename__}_document_id" in names
        assert f"ix_{model.__tablename__}_content_tsv" in names
