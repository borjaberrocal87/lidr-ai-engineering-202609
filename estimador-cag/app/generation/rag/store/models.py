"""Modelos ORM del store vectorial (Sesión 08 + multi-índice S10).

``documents`` es una fila por fuente ingerida (un presupuesto, una reunión, un
documento técnico) y posee la procedencia: de dónde vino, cuándo y su metadata.

El almacenamiento de chunks está **particionado por fuente** (Sesión 10):

* ``budget_chunks`` — la tabla ``chunks`` de la Sesión 08, renombrada en la
  migración ``0004``.
* ``transcript_chunks`` — segmentos de reuniones.
* ``technical_doc_chunks`` — secciones de documentación técnica.

Las tres comparten columnas (mixin ``_ChunkColumns``) pero su metadata JSONB
diverge: presupuestos llevan sector/año, transcripciones llevan hablantes y fecha
de reunión, y los documentos técnicos llevan versión. Esa divergencia es
justamente la razón de separar tablas: cada fuente tiene su propio índice
vectorial, su propio ciclo de vida y su propio esquema, en vez de una única tabla
donde el tipo dominante inunda el top-k.

Decisiones que se defienden en el README:

* ``metadata`` es JSONB en todas las tablas. Lo estable vive en columnas tipadas;
  lo que el chunker enriquece va a JSONB, consultable con GIN sin una migración
  por cada clave nueva.
* ``embedding`` es **nullable**: permite insertar el chunk y rellenar el vector
  después (ingesta asíncrona futura).
* ``Vector(1536)`` está hardcodeado a la dimensionalidad de
  ``text-embedding-3-small``. Cambiarlo implica re-embedear el corpus.
* ``content_tsv`` es un tsvector generado y almacenado que respalda la rama
  léxica. ``FTS_REGCONFIG`` debe coincidir con la migración y con la consulta
  léxica del repositorio, o Postgres ignorará el índice GIN en silencio.

``metadata`` es un atributo reservado en los modelos declarativos de SQLAlchemy,
así que el atributo Python es ``metadata_`` mapeado a la columna ``"metadata"``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Computed, ForeignKey, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship
from sqlalchemy.types import DateTime

from app.foundation.persistence.models import Base

EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small

# Configuración de búsqueda de texto de Postgres. Debe coincidir con la columna
# generada, con la migración y con ``websearch_to_tsquery`` del repositorio.
FTS_REGCONFIG = "english"


class DocumentRow(Base):
    """Una fuente ingerida: presupuesto, reunión o documento técnico."""

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

    # Solo los presupuestos mantienen la relación ORM bidireccional (herencia
    # S08/S09). Transcripciones y documentos confían en el ON DELETE CASCADE de
    # la base de datos para no tener que desambiguar tres back-references.
    chunks: Mapped[list[BudgetChunkRow]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class _ChunkColumns:
    """Columnas compartidas por las tres tablas de chunks.

    Es un mixin declarativo: SQLAlchemy copia estas columnas en cada tabla
    concreta. Los índices se generan a partir del ``__tablename__`` de cada
    subclase, de modo que las tres fuentes quedan estructuralmente idénticas sin
    repetir la lista de columnas tres veces.
    """

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
    # tsvector generado y almacenado: Postgres lo mantiene desde ``content`` (sin
    # triggers ni deriva). Solo lectura a nivel ORM.
    content_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(f"to_tsvector('{FTS_REGCONFIG}', content)", persisted=True),
        nullable=True,
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    @declared_attr.directive
    def __table_args__(cls) -> tuple:
        name = cls.__tablename__
        return (
            Index(f"ix_{name}_document_id", "document_id"),
            Index(f"ix_{name}_chunk_type", "chunk_type"),
            Index(f"ix_{name}_metadata_gin", "metadata", postgresql_using="gin"),
            Index(f"ix_{name}_content_tsv", "content_tsv", postgresql_using="gin"),
        )


class BudgetChunkRow(_ChunkColumns, Base):
    """Chunks de presupuestos históricos (la antigua tabla ``chunks``)."""

    __tablename__ = "budget_chunks"

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")


class TranscriptChunkRow(_ChunkColumns, Base):
    """Segmentos de reuniones. Su metadata lleva ``speakers`` y ``meeting_date``."""

    __tablename__ = "transcript_chunks"


class TechnicalDocChunkRow(_ChunkColumns, Base):
    """Secciones de documentación técnica. Su metadata lleva ``version``."""

    __tablename__ = "technical_doc_chunks"


# Alias de compatibilidad: el código S08/S09 importa ``ChunkRow`` y se refiere a
# la colección de presupuestos. Mantenerlo evita tocar cada import existente.
ChunkRow = BudgetChunkRow
