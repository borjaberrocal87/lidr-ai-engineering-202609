"""Modelos ORM del store vectorial (Sesión 08).

Dos tablas, un uno-a-muchos real:

* ``documents`` — una fila por fuente ingerida (un presupuesto histórico).
  Posee la procedencia: de dónde vino, cuándo y la metadata del documento.
* ``chunks`` — N filas por documento; cada una lleva el texto embebible y su
  vector de 1536 dimensiones. ``ON DELETE CASCADE`` elimina los chunks al
  borrar el presupuesto: integridad referencial en vez de duplicación.

Decisiones que se defienden en el README:

* ``metadata`` es JSONB en ambas tablas. Lo estable vive en columnas tipadas; lo
  que el chunker puede enriquecer va a JSONB, consultable con el índice GIN sin
  una migración por cada clave nueva.
* ``embedding`` es **nullable**: permite insertar el chunk y rellenar el vector
  después (ingesta asíncrona futura). En esta sesión se escriben chunk y
  embedding atómicamente.
* ``Vector(1536)`` está hardcodeado a la dimensionalidad de
  ``text-embedding-3-small``. Cambiarlo implica re-embedear el corpus; no es
  configuración dinámica.
* **No hay índice vectorial a propósito.** El sequential scan es el baseline
  que el directo mide contra HNSW.

``metadata`` es un atributo reservado en los modelos declarativos de
SQLAlchemy, así que el atributo Python es ``metadata_`` mapeado a la columna
``"metadata"``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, ForeignKey, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime

from app.persistence.models import Base

EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small


class DocumentRow(Base):
    """Un presupuesto histórico ingerido."""

    __tablename__ = "documents"
    __table_args__ = (Index("ix_documents_source_path", "source_path"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    chunks: Mapped[list[ChunkRow]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChunkRow(Base):
    """Un componente del presupuesto, con su embedding."""

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_chunk_type", "chunk_type"),
        Index("ix_chunks_metadata_gin", "metadata", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_type: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=True
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")
