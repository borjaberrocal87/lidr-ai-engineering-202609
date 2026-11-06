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
from sqlalchemy import Integer, Select, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.rag.schemas import EmbeddedChunk
from app.generation.rag.store.models import EMBEDDING_DIMENSIONS, ChunkRow, DocumentRow

# El chunker estructural emite un chunk por componente. El vocabulario es
# consultable gracias al índice sobre ``chunk_type`` (filtros del directo).
BUDGET_COMPONENT = "budget_component"

# Session 10: text-search configuration. MUST match the generated column in
# migration 0003. Change both places together if the corpus language changes.
FTS_CONFIG = "english"

# A lexical-only hit never entered the vector ranking, so it has no cosine
# distance. 1.0 is a display sentinel ("far"); fusion and reranking only use the
# ordering, never this distance.
LEXICAL_ONLY_DISTANCE = 1.0


def _or_tsquery_text(query_text: str) -> str:
    """Turn free text into an OR-of-terms query for ``websearch_to_tsquery``.

    ``plainto_tsquery`` joins every lexeme with AND, which is far too strict for
    long natural-language project descriptions: a single missing word drops the
    whole document. Keyword recall should surface documents that match *some*
    terms; RRF and the cross-encoder handle precision afterwards. Quoted phrase
    syntax is intentionally not supported here — the golden set uses plain text.
    """
    return " or ".join(query_text.split())


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
        chunk_type: str = BUDGET_COMPONENT,
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
                    chunk_type=chunk_type,
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

    async def search_with_filters(
        self,
        session: AsyncSession,
        *,
        query_vector: list[float],
        top_k: int = 10,
        distance_threshold: float = 0.6,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
    ) -> tuple[list[ChunkMatch], int]:
        """k-NN con pre-filtrado estructural y umbral de relevancia.

        Los filtros estructurales (sector, año, tipo de chunk) reducen el
        espacio de candidatos ANTES del ranking vectorial: la metadata viaja en
        JSONB (``client_sector``, ``year``) y el tipo en la columna
        ``chunk_type``. Cada filtro sigue el patrón "``None`` = no aplica".

        Devuelve ``(matches, candidates_evaluated)``: los ``top_k`` chunks bajo
        el umbral (distancia ascendente) y cuántos chunks pasaron los filtros
        estructurales antes del umbral y del límite.
        """
        structural_filters = self._structural_filters(
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
        )

        count_stmt = (
            select(func.count())
            .select_from(ChunkRow)
            .where(ChunkRow.embedding.is_not(None))
            .where(*structural_filters)
        )
        candidates_evaluated = int((await session.execute(count_stmt)).scalar_one())

        distance = self._distance_expression(query_vector)
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
            .where(*structural_filters)
            .where(distance <= distance_threshold)
            .order_by(distance)
            .limit(top_k)
        )
        rows = (await session.execute(stmt)).mappings().all()
        matches = [
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
        return matches, candidates_evaluated

    def _lexical_statement(
        self,
        *,
        query_text: str,
        top_k: int,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
    ) -> Select[Any]:
        """SELECT for the lexical branch: OR tsquery + ``ts_rank_cd``.

        ``websearch_to_tsquery`` receives an OR-of-terms string so a document can
        match *some* keywords instead of all of them. For long project
        descriptions, an AND query (``plainto_tsquery``) almost always returns
        zero rows, which makes the hybrid branch useless. ``ts_rank_cd`` (cover
        density) ranks by relevance: higher is better, the opposite of cosine
        distance. ``ORDER BY ... DESC`` reflects that.
        """
        tsquery = func.websearch_to_tsquery(FTS_CONFIG, _or_tsquery_text(query_text))
        rank = func.ts_rank_cd(ChunkRow.content_tsv, tsquery)
        return (
            select(
                ChunkRow.id.label("chunk_id"),
                ChunkRow.document_id.label("document_id"),
                ChunkRow.chunk_type.label("chunk_type"),
                ChunkRow.content.label("content"),
                ChunkRow.metadata_.label("metadata_"),
                rank.label("rank"),
            )
            .where(ChunkRow.content_tsv.op("@@")(tsquery))
            .where(
                *self._structural_filters(
                    sectors=sectors,
                    project_year_min=project_year_min,
                    project_year_max=project_year_max,
                    chunk_types=chunk_types,
                )
            )
            .order_by(rank.desc(), ChunkRow.id.asc())
            .limit(top_k)
        )

    async def search_lexical(
        self,
        session: AsyncSession,
        *,
        query_text: str,
        top_k: int = 50,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
    ) -> list[ChunkMatch]:
        """Lexical branch of hybrid search (Session 10).

        Uses the same structural filters as ``search_with_filters`` so both
        branches see the same candidate space. Returns at most ``top_k`` chunks
        ordered by ``ts_rank_cd`` descending.
        """
        stmt = self._lexical_statement(
            query_text=query_text,
            top_k=top_k,
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
        )
        rows = (await session.execute(stmt)).mappings().all()
        return [
            ChunkMatch(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                chunk_type=row["chunk_type"],
                content=row["content"],
                distance=LEXICAL_ONLY_DISTANCE,
                metadata=row["metadata_"],
            )
            for row in rows
        ]

    @staticmethod
    def _structural_filters(
        *,
        sectors: list[str] | None,
        project_year_min: int | None,
        project_year_max: int | None,
        chunk_types: list[str] | None,
    ) -> list[Any]:
        """Predicates shared by the vector and lexical branches.

        Each filter follows the ``None`` = do not apply convention. Sector and
        year read from JSONB; chunk type reads from its own column.
        """
        sector_col = ChunkRow.metadata_["client_sector"].astext
        year_col = cast(ChunkRow.metadata_["year"].astext, Integer)

        filters: list[Any] = []
        if sectors:
            filters.append(sector_col.in_(sectors))
        if project_year_min is not None:
            filters.append(year_col >= project_year_min)
        if project_year_max is not None:
            filters.append(year_col <= project_year_max)
        if chunk_types:
            filters.append(ChunkRow.chunk_type.in_(chunk_types))
        return filters
