"""Sesión 11 (directo) — índices HNSW sobre las tres colecciones de chunks.

Revision ID: 0005_session11_hnsw_multi_index
Revises: 0004_session10_multi_index
Create Date: 2026-11-19 00:00:00

La 0004 dejó las tres tablas de chunks (``budget_chunks``, ``transcript_chunks``
y ``technical_doc_chunks``) a sequential scan. Con la expansión incremental de la
Sesión 11 el corpus PUEDE crecer, así que el scan se vuelve linealmente más lento:
esta migración provisiona el índice vectorial y, sobre todo, lo hace de forma que
Postgres lo USE.

El retriever busca casteando el vector almacenado a ``halfvec`` y operando con
distancia coseno (ver ``store/repository.py``: ``cast(embedding, HALFVEC(1536))
.cosine_distance(...)``). Un índice HNSW solo se consulta si su operator class
coincide con el operador de la query: un índice ``vector_cosine_ops`` (el script
manual de la Sesión 08) queda **ignorado en silencio** por una búsqueda halfvec.
Por eso el índice se construye sobre la MISMA expresión
``embedding::halfvec(1536)`` con ``halfvec_cosine_ops``: mitad de tamaño y
coincidencia exacta con el camino de búsqueda.

Con el índice creado, HNSW se mantiene INCREMENTALMENTE: cada chunk que inserta
la expansión del corpus entra en el grafo sin reconstruirlo.

``m=16`` / ``ef_construction=128`` replican las decisiones del directo de la
Sesión 08. ``IF NOT EXISTS`` lo hace idempotente ante reejecuciones al cambiar de
rama.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0005_session11_hnsw_multi_index"
down_revision: str | None = "0004_session10_multi_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CHUNK_TABLES = ("budget_chunks", "transcript_chunks", "technical_doc_chunks")
EMBEDDING_DIMENSIONS = 1536
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 128


def _index_name(table: str) -> str:
    # Debe coincidir con el nombre que consulta ``ChunkStore.corpus_stats``.
    return f"ix_{table}_embedding_hnsw"


def upgrade() -> None:
    for table in CHUNK_TABLES:
        op.execute(
            f"CREATE INDEX IF NOT EXISTS {_index_name(table)} "
            f"ON {table} "
            f"USING hnsw ((embedding::halfvec({EMBEDDING_DIMENSIONS})) halfvec_cosine_ops) "
            f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
        )


def downgrade() -> None:
    for table in CHUNK_TABLES:
        op.execute(f"DROP INDEX IF EXISTS {_index_name(table)}")
