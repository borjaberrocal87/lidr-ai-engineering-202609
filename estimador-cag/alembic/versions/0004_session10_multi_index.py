"""Sesión 10 — multi-índice: renombrar chunks y añadir transcripciones y docs.

Revision ID: 0004_session10_multi_index
Revises: 0003_session10_fts
Create Date: 2026-11-13 00:00:00

La recuperación multifuente necesita que cada fuente tenga su propia tabla: su
esquema de metadata, su índice vectorial y su ciclo de vida. Esta migración:

1. Renombra la tabla ``chunks`` de la Sesión 08 a ``budget_chunks`` y renombra
   sus índices. La columna generada ``content_tsv`` viaja con el rename, sin
   recrearse.
2. Crea ``transcript_chunks`` y ``technical_doc_chunks`` con las mismas columnas
   estructurales (comparten el mixin ``_ChunkColumns`` en el ORM) pero con su
   propia metadata JSONB, sus índices GIN y su ``content_tsv``.

La configuración de búsqueda de texto sigue siendo ``english`` en todas las
tablas (el corpus canónico está en inglés; ver la nota de la migración 0003).

No se crea índice vectorial (HNSW) aquí a propósito: los índices vectoriales son
el tema del directo de la Sesión 08 y viven en un script SQL manual, no en
Alembic. Las tablas nuevas arrancan con sequential scan, igual que arrancó la
tabla de presupuestos.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0004_session10_multi_index"
down_revision: str | None = "0003_session10_fts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FTS_REGCONFIG = "english"
NEW_COLLECTIONS = ("transcript_chunks", "technical_doc_chunks")

# Índices de la antigua tabla ``chunks`` (nombre viejo, nombre nuevo).
_BUDGET_INDEX_RENAMES = (
    ("ix_chunks_document_id", "ix_budget_chunks_document_id"),
    ("ix_chunks_chunk_type", "ix_budget_chunks_chunk_type"),
    ("ix_chunks_metadata_gin", "ix_budget_chunks_metadata_gin"),
    ("ix_chunks_content_tsv", "ix_budget_chunks_content_tsv"),
)


def _create_chunk_table(table: str) -> None:
    """Crea una tabla de colección con la misma forma que ``budget_chunks``."""
    op.create_table(
        table,
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "document_id",
            sa.BigInteger,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_type", sa.String(length=50), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    # tsvector generado: DDL en crudo porque Alembic no tiene helper de primera
    # clase para columnas generadas.
    op.execute(
        f"ALTER TABLE {table} ADD COLUMN content_tsv tsvector "
        f"GENERATED ALWAYS AS (to_tsvector('{FTS_REGCONFIG}', content)) STORED"
    )
    op.create_index(f"ix_{table}_document_id", table, ["document_id"])
    op.create_index(f"ix_{table}_chunk_type", table, ["chunk_type"])
    op.create_index(f"ix_{table}_metadata_gin", table, ["metadata"], postgresql_using="gin")
    op.create_index(f"ix_{table}_content_tsv", table, ["content_tsv"], postgresql_using="gin")


def _drop_chunk_table(table: str) -> None:
    op.drop_index(f"ix_{table}_content_tsv", table_name=table)
    op.drop_index(f"ix_{table}_metadata_gin", table_name=table)
    op.drop_index(f"ix_{table}_chunk_type", table_name=table)
    op.drop_index(f"ix_{table}_document_id", table_name=table)
    op.drop_table(table)


def upgrade() -> None:
    # 1. Renombrar la tabla de presupuestos y sus índices a su nombre de fuente.
    op.rename_table("chunks", "budget_chunks")
    for old, new in _BUDGET_INDEX_RENAMES:
        op.execute(f"ALTER INDEX {old} RENAME TO {new}")

    # 2. Crear las dos colecciones nuevas.
    for table in NEW_COLLECTIONS:
        _create_chunk_table(table)


def downgrade() -> None:
    for table in NEW_COLLECTIONS:
        _drop_chunk_table(table)

    for old, new in _BUDGET_INDEX_RENAMES:
        op.execute(f"ALTER INDEX {new} RENAME TO {old}")
    op.rename_table("budget_chunks", "chunks")
