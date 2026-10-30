"""Tests de los modelos ORM del store vectorial (documents/chunks)."""

from __future__ import annotations

from app.embedding_pipeline.store.models import (
    EMBEDDING_DIMENSIONS,
    ChunkRow,
    DocumentRow,
)
from app.persistence.models import Base


def test_store_tables_are_registered_in_the_shared_base() -> None:
    assert DocumentRow.__tablename__ == "documents"
    assert ChunkRow.__tablename__ == "chunks"
    assert DocumentRow.metadata.tables["documents"] is DocumentRow.__table__
    assert DocumentRow.metadata.tables["chunks"] is ChunkRow.__table__
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
    }
