"""Session 10 — full-text column (tsvector) and GIN index on chunks.

Revision ID: 0003_session10_fts
Revises: 0002_session8_pgvector
Create Date: 2026-11-06 00:00:00

Adds ``content_tsv``, a STORED generated column derived from ``content``, plus
its GIN index. It backs the lexical branch of hybrid search: the vector branch
keeps using ``embedding`` and both rankings are fused with RRF.

Text-search configuration:
- The exercise statement says the corpus is in Spanish, but this repository's
  canonical corpus (``data/budgets_sample.json``, ``data/task_corpus.json``) is
  in English. We use ``english`` because its stemming and stop words are what
  actually lift lexical recall on this data.
- If a Spanish corpus is ingested later, change ``english`` to ``spanish`` in
  TWO places: this migration and ``plainto_tsquery`` in the repository. The
  generated column cannot change without being recreated.

The column is GENERATED ALWAYS ... STORED: Postgres recomputes it on every
insert/update of ``content``, so there is no trigger to maintain and no way for
the lexical index to drift from the text it indexes.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003_session10_fts"
down_revision: str | None = "0002_session8_pgvector"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Raw DDL is the clearest expression of a STORED generated tsvector:
    # Alembic/SQLAlchemy have no first-class helper for it.
    op.execute(
        "ALTER TABLE chunks ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    op.create_index(
        "ix_chunks_content_tsv",
        "chunks",
        ["content_tsv"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_content_tsv", table_name="chunks")
    op.drop_column("chunks", "content_tsv")
