"""Capa de acceso a datos async del store vectorial.

El store no abre ni confirma sesiones: el llamante (servicio de ingesta,
retriever) es dueño de la ``AsyncSession``. Así una ingesta completa —
chequeo de duplicado, documento y chunks — cabe en UNA transacción. Un fallo
en cualquier punto hace rollback y no deja ``documents`` huérfanos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding_pipeline.schemas import EmbeddedChunk
from app.embedding_pipeline.store.models import ChunkRow, DocumentRow

# El chunker estructural emite un chunk por componente. El vocabulario es
# consultable gracias al índice sobre ``chunk_type`` (filtros del directo).
BUDGET_COMPONENT = "budget_component"


@dataclass(frozen=True, slots=True)
class ChunkMatch:
    """Una fila de resultado ya desacoplada del ``Row`` de SQLAlchemy."""

    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float
    metadata: dict[str, Any]


class ChunkStore:
    """CRUD y búsqueda por similitud sobre ``documents``/``chunks``."""

    async def find_document_id(self, session: AsyncSession, source_path: str) -> int | None:
        """Id del documento ya ingerido desde ``source_path``, o ``None``.

        Es el soporte del ``409`` a nivel de aplicación. El ``limit(1)`` evita
        ``MultipleResultsFound`` si una carrera histórica dejó duplicados; el
        chequeo sigue sin ser race-proof.
        """
        stmt = select(DocumentRow.id).where(DocumentRow.source_path == source_path).limit(1)
        return (await session.execute(stmt)).scalar_one_or_none()

    async def persist_document_with_chunks(
        self,
        session: AsyncSession,
        *,
        source_path: str,
        document_type: str,
        doc_metadata: dict[str, Any],
        embedded_chunks: list[EmbeddedChunk],
    ) -> int:
        """Inserta el documento y todos sus chunks. Sin commit.

        El ``flush`` asigna ``document.id`` sin cerrar la transacción; los
        chunks se añaden con ``add_all`` y el commit lo decide el llamante.
        """
        document = DocumentRow(
            source_path=source_path,
            document_type=document_type,
            metadata_=doc_metadata,
        )
        session.add(document)
        await session.flush()  # asigna document.id sin commitear

        session.add_all(
            [
                ChunkRow(
                    document_id=document.id,
                    chunk_type=BUDGET_COMPONENT,
                    content=chunk.text,
                    embedding=chunk.embedding,
                    metadata_=chunk.metadata,
                )
                for chunk in embedded_chunks
            ]
        )
        return document.id

    async def search(
        self,
        session: AsyncSession,
        *,
        query_vector: list[float],
        k: int,
    ) -> list[ChunkMatch]:
        """Los ``k`` chunks más cercanos por distancia coseno (``<=>``).

        Coseno frente a L2 o producto interno: los embeddings de OpenAI están
        normalizados, así que el ranking sería equivalente; usamos coseno por
        convención RAG y, sobre todo, para quedar alineados con la operator
        class ``vector_cosine_ops`` del índice HNSW que se añade en el directo.
        Un operador y una operator class distintos hacen que Postgres ignore el
        índice en silencio.
        """
        distance = ChunkRow.embedding.cosine_distance(query_vector)
        stmt = (
            select(
                ChunkRow.id.label("chunk_id"),
                ChunkRow.document_id.label("document_id"),
                ChunkRow.chunk_type.label("chunk_type"),
                ChunkRow.content.label("content"),
                ChunkRow.metadata_.label("metadata_"),
                distance.label("distance"),
            )
            .where(ChunkRow.embedding.is_not(None))
            .order_by(distance)
            .limit(k)
        )
        rows = (await session.execute(stmt)).mappings().all()
        return [
            ChunkMatch(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                chunk_type=row["chunk_type"],
                content=row["content"],
                distance=float(row["distance"]),
                metadata=row["metadata_"],
            )
            for row in rows
        ]
