"""Sesión 08 — extensión pgvector + tablas documents/chunks.

Revision ID: 0002_session8_pgvector
Revises: 0001_session6_initial
Create Date: 2026-06-04 00:00:00

Deliberadamente SIN índice vectorial (HNSW / IVFFlat): el sequential scan es el
baseline contra el que el directo mide el impacto del índice. Los índices no
vectoriales (FK, chunk_type y GIN sobre metadata) SÍ se crean aquí porque son
higiene relacional, no el objeto de la demo.

``ix_documents_source_path`` es un índice plano (no único), literal al
enunciado; la unicidad de ``source_path`` la impone el chequeo de aplicación
que devuelve 409. Ese check-then-insert no es race-proof bajo ingestas
idénticas concurrentes — aceptable a escala de formación y buen gancho para el
directo.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002_session8_pgvector"
down_revision: str | None = "0001_session6_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Debe ejecutarse antes de que exista cualquier columna Vector. La imagen
    # pgvector/pgvector:pg16 trae la extensión compilada; esta sentencia la
    # activa.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("source_path", sa.Text, nullable=False),
        sa.Column("document_type", sa.String(length=50), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_index("ix_documents_source_path", "documents", ["source_path"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "document_id",
            sa.BigInteger,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_type", sa.String(length=50), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        # Nullable: deja abierta la puerta a insertar el chunk y rellenar el
        # vector después (ingesta asíncrona futura). En esta sesión se escriben
        # chunk + embedding atómicamente.
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
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_chunk_type", "chunks", ["chunk_type"])
    # GIN sobre el JSONB completo: consultar claves arbitrarias sin una
    # migración por cada clave nueva.
    op.create_index(
        "ix_chunks_metadata_gin",
        "chunks",
        ["metadata"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_metadata_gin", table_name="chunks")
    op.drop_index("ix_chunks_chunk_type", table_name="chunks")
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_documents_source_path", table_name="documents")
    op.drop_table("documents")
    # La extensión se deja instalada: borrarla es una operación de clúster que
    # podría romper otros objetos. Recrearla es gratis (IF NOT EXISTS).
