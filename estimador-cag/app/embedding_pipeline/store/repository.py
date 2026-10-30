"""Capa de acceso a datos async del store vectorial.

El store no abre ni confirma sesiones: el llamante (servicio de ingesta,
retriever) es dueño de la ``AsyncSession``. Así una ingesta completa —
chequeo de duplicado, documento y chunks — cabe en UNA transacción. Un fallo
en cualquier punto hace rollback y no deja ``documents`` huérfanos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import Select, cast, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding_pipeline.schemas import EmbeddedChunk
from app.embedding_pipeline.store.models import EMBEDDING_DIMENSIONS, ChunkRow, DocumentRow

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
    """CRUD y búsqueda por similitud sobre ``documents``/``chunks``.

    ``use_halfvec`` activa la variante de media precisión para la distancia:
    la columna sigue siendo ``vector(1536)``, pero la query castea la expresión
    a ``halfvec(1536)`` para que Postgres pueda usar el índice HNSW construido
    sobre ``(embedding::halfvec(1536))``. Está apagado por defecto porque, sin
    ese índice, el cast por fila puede empeorar el plan.
    """

    def __init__(self, *, use_halfvec: bool = False) -> None:
        self._use_halfvec = use_halfvec

    @property
    def search_mode(self) -> str:
        """Nombre del modo de búsqueda, útil para logs y benchmarks."""
        return "halfvec" if self._use_halfvec else "vector"

    def _distance_expression(self, query_vector: list[float]) -> Any:
        """Expresión de distancia coseno alineada con el índice activo."""
        if self._use_halfvec:
            return cast(
                ChunkRow.embedding,
                HALFVEC(EMBEDDING_DIMENSIONS),
            ).cosine_distance(query_vector)
        return ChunkRow.embedding.cosine_distance(query_vector)

    def _search_statement(self, query_vector: list[float], k: int) -> Select[Any]:
        """SELECT k-NN reutilizable por la API y por los benchmarks."""
        distance = self._distance_expression(query_vector)
        return (
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
        stmt = self._search_statement(query_vector, k)
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
