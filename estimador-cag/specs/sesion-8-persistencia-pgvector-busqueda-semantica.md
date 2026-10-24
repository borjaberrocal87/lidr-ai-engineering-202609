# Spec — Proyecto 2: Persistencia vectorial y búsqueda semántica (PostgreSQL + pgvector)

## Objetivo

La Sesión 07 cerró con un pipeline de embeddings funcional, pero **efímero**: cada llamada a `POST /api/v1/embeddings/ingest` volvía a chunkear, volvía a embeber y devolvía los vectores por HTTP. El corpus no existía en ningún sitio después de la respuesta. Esta sesión convierte ese pipeline en un **sistema con memoria**: los presupuestos históricos se persisten en PostgreSQL con pgvector y se consultan por similitud semántica desde SQL.

Al terminar este ejercicio tendrás:

- Un esquema relacional propio (`documents` y `chunks`) gestionado con **Alembic**, conviviendo con las tablas de la Sesión 06.
- El endpoint **`POST /api/v1/embeddings/ingest`** refactorizado: un documento por petición, persistido con sus chunks y embeddings en **una sola transacción**, con `409` para documentos ya ingeridos.
- El endpoint **`POST /api/v1/search`**, que embebe la query con el mismo modelo de la ingesta y devuelve los `k` chunks más cercanos por **distancia coseno** (`<=>`).
- Un script **`scripts/query_examples.py`** que ingesta el corpus de ejemplo de forma idempotente y lanza cinco queries representativas.
- La evidencia ejecutada en **`data/output/session08/output_examples.txt`**.
- Una sección nueva en el README, de máximo una página, con las cuatro justificaciones de diseño.

> **Reconciliación con el repo.** El punto de partida es `main` con las sesiones 01–07 cerradas: wrapper LLM, cachés, sesiones conversacionales, ingesta con PII, baseline de estrés, pipeline de embeddings y comparativa de estrategias de chunking. **Nada de eso se reescribe.** En particular:
>
> - El servicio `estimator-postgres` **ya existe** en `docker-compose.yml` con la imagen `pgvector/pgvector:pg16` desde la Sesión 06. No se añade un segundo Postgres ni un servicio `postgres` nuevo.
> - La URL de conexión **ya existe** como `DATABASE_URL` en `.env` y se mapea a `settings.database_url` (snake_case) en `app/config.py`. No hay variable de entorno nueva.
> - La Sesión 06 usa SQLAlchemy **síncrono** con `psycopg` para el tracking de jobs y el mapping de PII. Ese stack se queda intacto. La Sesión 08 añade un **stack async con `asyncpg`** solo para la ruta caliente de ingest/search.
> - Ya hay una migración (`0001_session6_initial`), así que la nueva es **`0002_session8_pgvector`**, no una migración inicial.
> - El contrato actual del ingest (`{"budgets": [...]}` → chunks + vectores) **se reemplaza**. El endpoint `/embeddings/compare` y el Chunking Lab del cliente de negocio no se tocan.
> - El módulo vive en **`app/embedding_pipeline/`**, la casa que ya tiene el chunker, el embedder y los contratos. No introducimos un `app/generation/rag/` ajeno a la estructura del repo.

## Contexto del proyecto

Piezas relevantes que se reutilizan o extienden (todas ya en `main`):

- `app/embedding_pipeline/structural.py` — `JSONStructuralChunker`: un componente del presupuesto = un chunk, con cabecera de contexto y `token_count`. Es el chunker que usa la ingesta persistente.
- `app/embedding_pipeline/schemas.py` — `Budget`, `Chunk`, `EmbeddedChunk`, `IngestRequest`, `IngestResponse`. Aquí se sustituye el contrato viejo y se añaden los modelos de búsqueda. `Sector` y `Complexity` son `StrEnum`.
- `app/embedding_pipeline/embedder.py` — `OpenAIEmbedder` (síncrono) con batches de 100, reintentos, `embed_one` para queries y `from_settings` como única fábrica del cliente. No se toca.
- `app/persistence/database.py` — motor síncrono, `SessionLocal` y `get_session()` de la Sesión 06. Se le añade, al lado, el motor async y su `async_sessionmaker`.
- `app/persistence/models.py` — `Base` declarativa única. Los modelos de `documents`/`chunks` se registran contra esta misma `Base` para que Alembic los descubra.
- `alembic/env.py` y `alembic/versions/0001_session6_initial.py` — entorno de migraciones síncrono y primera migración. Se extienden con el tipo `vector` y la migración `0002`.
- `app/dependencies.py` — fábricas `lru_cache` (`get_chunker`, `get_embedder`, `get_session_store`…). Aquí se añaden `get_chunk_store`, `get_rag_ingest_service` y `get_semantic_retriever`, con la misma política de degradación a `None`.
- `app/main.py` — los routers se registran con prefijo `/api/v1`. El router de búsqueda se suma ahí: la ruta completa es `/api/v1/search`.
- `app/routers/` — routers HTTP por recurso (`estimations`, `sessions`, `ingestion`, `config`). `search.py` se añade aquí porque la búsqueda no es un subrecurso de `/embeddings`.
- `docker-compose.yml` — `api` depende de `estimator-postgres` con `condition: service_healthy`; el `command` del servicio aplica `alembic upgrade head` al arrancar. No hay que tocar Compose.
- `Dockerfile` — copia `data/`, `alembic/` y `scripts/` a la imagen. El script nuevo viaja dentro sin cambios.
- `pyproject.toml` — `sqlalchemy>=2.0`, `alembic>=1.13`, `psycopg[binary]` y `httpx` ya están. Solo faltan `asyncpg`, `pgvector` y `greenlet` (requisito de la extensión asyncio de SQLAlchemy).
- `tests/conftest.py` — fixture autouse que limpia las cachés `lru_cache` entre tests. Hay que añadir las fábricas nuevas.
- `tests/test_embedding_endpoint.py`, `tests/test_embedding_chunker.py` y `tests/test_project_structure.py` — tests que el contrato nuevo obliga a actualizar.

## Requisitos para el ejercicio

- Proyecto de la sesión 07 en verde:

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy app frontend
  ```

- Dependencias nuevas:

  ```bash
  uv add asyncpg pgvector greenlet
  ```

  `sqlalchemy>=2.0` y `alembic>=1.13` ya están en `pyproject.toml`. No añadas `psycopg` otra vez: el stack síncrono de la Sesión 06 sigue usándolo.

- Postgres ya levantado (imagen pgvector, puerto host `5433`):

  ```bash
  docker compose up -d estimator-postgres
  docker compose exec estimator-postgres psql -U estimator -d estimator -c "SELECT version();"
  ```

  Si esto no funciona, no avances al Paso 2.

- API key de OpenAI configurada en `.env` (`OPEN_AI_KEY`) o un endpoint compatible con embeddings (`EMBEDDING_BASE_URL` + `EMBEDDING_API_KEY`). Sin embeddings, el ingest y la búsqueda responden `500` con mensaje genérico.

- Docker y Docker Compose funcionando.

- `uv.lock` actualizado y commiteado: el `Dockerfile` construye con `uv sync --frozen`.

> **No se añade un servicio `postgres` nuevo.** El enunciado original proponía uno; este repo ya tiene `estimator-postgres` con pgvector desde la Sesión 06 y una base de negocio distinta (`postgres`) para `cliente-web`. Mezclar ambas sería un error de aislamiento. La decisión se documenta en la sección de la Sesión 08 del README.

## Lo que entra y lo que no entra

**Entra:**

- Dependencias `asyncpg`, `pgvector` y `greenlet`, y un motor async para la ruta RAG.
- Modelos ORM `DocumentRow` y `ChunkRow` contra la `Base` existente.
- Migración Alembic `0002_session8_pgvector` con la extensión `vector`, las dos tablas y los índices **no vectoriales**.
- Repositorio async `ChunkStore` con verificación de duplicado, persistencia transaccional y búsqueda por coseno.
- Servicio de ingesta `RagIngestService` y retriever `SemanticRetriever`.
- Refactor de `POST /api/v1/embeddings/ingest` y nuevo `POST /api/v1/search`.
- Script `scripts/query_examples.py` y evidencia en `data/output/session08/output_examples.txt`.
- Sección nueva del README y actualización de la sección de la Sesión 07.
- Tests HTTP con dobles y actualización del test de estructura.

**No entra** (se aborda deliberadamente en directo):

- Índices vectoriales (`HNSW`, `IVFFlat`). El **sequential scan** es el baseline contra el que se medirá el impacto del índice.
- Filtros por metadata en SQL (`WHERE chunk_type = 'budget_component'`, `WHERE metadata->>'sector' = 'fintech'`). Se exploran en directo comparando planes de ejecución.
- Búsqueda híbrida (full-text search + vector). Se construye en directo sobre este código.
- Tuning de parámetros (`shared_buffers`, `maintenance_work_mem`, `ef_search`). Todo con defaults.
- Comparativa de modelos de embeddings, Matryoshka persistido o re-embedding masivo.
- Cambios en la UI, en `cliente-web` o en el endpoint `/embeddings/compare`.
- Métricas formales de retrieval (`recall@k`, NDCG) y tests de integración con una base de datos viva.

## Datos para el ejercicio

Se trabaja con `data/budgets_sample.json`: **17 presupuestos** y **60 componentes**. Cada componente produce un chunk. El contrato nuevo ingesta **un documento por petición**, así que el script recorre el array y hace 17 llamadas.

Convención de `source_path` para el corpus de ejemplo:

```text
data/budgets_sample.json::BUD-2024-001
data/budgets_sample.json::BUD-2024-002
...
```

Ese sufijo es lo que hace único cada presupuesto dentro del mismo fichero y lo que permite que el `409` sea idempotente al reejecutar el script.

- `document_type`: `"historical_budget"`.
- `chunk_type`: `"budget_component"`.
- `metadata` del documento: `budget_id`, `client_sector`, `year`.
- `metadata` del chunk: la que ya produce `JSONStructuralChunker` (`budget_id`, `component_id`, `client_sector`, `main_technology`, `year`, `complexity`, `estimated_hours`).

## ✍ Ejercicio

### Paso 1 — Dependencias y convivencia sync / async

Añade las dos dependencias:

```bash
uv add asyncpg pgvector greenlet
```

`greenlet` no se usa directamente: es la dependencia que exige la extensión
`sqlalchemy.ext.asyncio`. Sin ella, importar `AsyncSession` falla al arrancar.

El repo tiene dos stacks a partir de ahora, y es **deliberado**:

- **Sync (`psycopg`)** — Sesión 06. Ingesta en `BackgroundTasks`, jobs administrativos y migraciones Alembic. No está en la ruta caliente del usuario.
- **Async (`asyncpg`)** — Sesión 08. `POST /embeddings/ingest` y `POST /api/v1/search` sí están en la ruta caliente: no pueden bloquear el event loop esperando a Postgres.

Ambos leen la misma `DATABASE_URL`. El motor async **deriva** el driver; no se introduce una segunda variable de entorno.

En `app/persistence/database.py`, junto al motor síncrono existente:

```python
"""Motor de SQLAlchemy, factorías de sesión y ayudantes de sesión por petición.

Dos stacks conviven a propósito:

* **Sync (psycopg)** — la ingesta de la Sesión 06. No está en la ruta caliente
  de peticiones de usuario, así que cambiamos ergonomía asíncrona por
  simplicidad.
* **Async (asyncpg)** — el store RAG de la Sesión 08. ``POST
  /embeddings/ingest`` y ``POST /search`` sí están en la ruta caliente y no
  deben bloquear el event loop en I/O de Postgres.

Ambos motores leen ``Settings.database_url``; el async cambia el token del
driver, de modo que una sola variable configura todo el servicio.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

# ... create_engine_from_settings(), SessionLocal y get_session() se quedan
# exactamente como están ...


def _async_database_url() -> str:
    """Deriva la URL de asyncpg desde ``Settings.database_url``.

    La URL canónica usa el driver síncrono (``postgresql+psycopg://``) porque
    Alembic y los repositorios de la Sesión 06 son síncronos. El store RAG
    cambia el token del driver en lugar de introducir una segunda variable.
    """
    url = get_settings().database_url
    if "+asyncpg" in url:
        return url
    if "+psycopg" in url:
        return url.replace("+psycopg", "+asyncpg")
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


@lru_cache
def create_async_engine_from_settings() -> AsyncEngine:
    """Motor async global para el store RAG (singleton)."""
    return create_async_engine(
        _async_database_url(),
        pool_pre_ping=True,
    )


@lru_cache
def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Factoría de sesiones async.

    ``expire_on_commit=False`` para que el ``document_id`` recién persistido
    siga siendo legible después del commit, sin un SELECT extra.
    """
    return async_sessionmaker(
        bind=create_async_engine_from_settings(),
        autoflush=False,
        expire_on_commit=False,
    )
```

> **No se toca `SessionLocal`.** Cambiar el stack de la Sesión 06 a async obligaría a reescribir repositorios y tests ya cerrados, y no aporta nada a este ejercicio. La ruta RAG tiene su propio motor.

> **El motor async es perezoso.** `create_async_engine_from_settings` solo se ejecuta cuando alguien pide la fábrica; importar `app.persistence.database` no abre conexiones ni exige un Postgres vivo. Eso mantiene los tests unitarios actuales sin infraestructura.

> **`async_sessionmaker[AsyncSession]`.** `mypy` en modo estricto no admite el genérico desnudo. La anotación explícita documenta el tipo de sesión que devuelve la factoría.

### Paso 2 — Modelos ORM del store vectorial

Crea el paquete `app/embedding_pipeline/store/`:

```text
app/embedding_pipeline/
└── store/
    ├── __init__.py
    ├── models.py
    └── repository.py
```

`__init__.py` es un docstring de módulo que resume la pieza y su alcance: persistencia vectorial mínima; sin índice vectorial, sin filtros y sin búsqueda híbrida.

En `store/models.py`:

```python
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
```

> **La `Base` es la de la Sesión 06.** `app.persistence.models.Base` ya es la base declarativa única que recoge `alembic/env.py`. Crear una segunda `Base` rompería el descubrimiento de modelos y obligaría a mantener dos `target_metadata`.

> **`metadata_` y no `metadata`.** Es la colisión clásica de SQLAlchemy: `metadata` está reservado por la API declarativa. El nombre físico de la columna sigue siendo `metadata`, que es el que espera el contrato SQL del ejercicio y el que se consulta con `->>`.

> **Sin índice único sobre `source_path`.** El enunciado pide un índice plano y un `409` a nivel de aplicación. La unicidad estricta exigiría una `UniqueConstraint`; se deja para el directo, donde se discute la carrera entre dos ingestas idénticas concurrentes.

### Paso 3 — Repositorio async

En `store/repository.py` implementa `ChunkStore`. Regla de oro: **el repositorio nunca abre ni hace commit de una sesión**. La transacción la posee el servicio; así el chequeo de duplicado, la fila del documento y las filas de chunks caen en la misma unidad atómica.

```python
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
```

> **`ChunkMatch` y no un `Row` crudo.** El repositorio es la frontera entre SQLAlchemy y el dominio. Devolver un `Row` obliga al retriever a conocer nombres de columnas y complica el tipado estricto. Un dataclass inmutable documenta el contrato y hace trivial el doble de tests.

> **`embedding.is_not(None)`.** La columna es nullable por diseño. Un chunk sin vector no puede participar en un ranking por distancia; filtrarlo evita filas con `distance = NULL` al final del `ORDER BY`.

> **Sin `commit()` en el repositorio.** Si el embedder falla después de insertar el documento, la transacción del servicio hace rollback. Ese es el punto entero del ejercicio.

### Paso 4 — Migración Alembic y registro del tipo `vector`

Crea `alembic/versions/0002_session8_pgvector.py`. El `down_revision` apunta a la migración de la Sesión 06:

```python
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
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

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
```

Actualiza `alembic/env.py`:

```python
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql.base import ischema_names

from app.persistence.models import Base
import app.embedding_pipeline.store.models  # noqa: F401 — registra S8 en Base.metadata

# ...

# Enseña a la reflexión el tipo ``vector``. Sin esto, ``alembic check`` /
# autogenerate contra una BD que ya tiene columnas vector no sabe mapearlas de
# vuelta y produce diffs inconsistentes.
ischema_names["vector"] = Vector

target_metadata = Base.metadata
```

> **Alembic sigue siendo síncrono.** Usa `psycopg` y la URL `postgresql+psycopg://`. El motor async es solo de runtime. No hay que duplicar `env.py` ni convertir las migraciones a async.

> **La extensión no se borra en `downgrade`.** Es una operación a nivel de clúster y podría afectar a otros objetos. La migración de bajada elimina tablas e índices; la extensión se queda.

Ejecuta la migración:

```bash
docker compose build api
docker compose run --rm api alembic upgrade head
```

> El servicio `api` ya aplica `alembic upgrade head` al arrancar. El comando anterior sirve para ejecutarla sin levantar el servidor. Si la imagen no se reconstruye, el contenedor no verá el código ni la migración nuevos.

Verifica el esquema:

```bash
docker compose exec estimator-postgres psql -U estimator -d estimator -c "\dx vector"
docker compose exec estimator-postgres psql -U estimator -d estimator -c "\dt documents"
docker compose exec estimator-postgres psql -U estimator -d estimator -c "\dt chunks"
```

### Paso 5 — Contratos HTTP: ingesta y búsqueda

En `app/embedding_pipeline/schemas.py`, conserva `Sector`, `Complexity`, `ClientMetadata`, `BudgetComponent`, `Budget`, `Chunk` y `EmbeddedChunk`. **Sustituye** `IngestRequest`, `IngestResponse` e `IngestStats` por los contratos de la sesión, y añade los de búsqueda:

```python
class IngestRequest(BaseModel):
    """Payload de ``POST /api/v1/embeddings/ingest`` (contrato persistente).

    Una petición = un documento. ``content`` es el JSON completo del
    presupuesto, validado contra :class:`Budget`, de modo que un corpus mal
    formado falla con 422 antes de tocar la base de datos o la API de
    embeddings.
    """

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Procedencia del documento; única por ingesta.",
    )
    document_type: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Familia documental, p. ej. 'historical_budget'.",
    )
    content: Budget = Field(..., description="JSON completo del presupuesto.")


class IngestResponse(BaseModel):
    """Respuesta de ``POST /api/v1/embeddings/ingest``.

    Los vectores ya no viajan por HTTP: viven en pgvector. Solo se devuelven
    identificadores y métricas de la ingesta.
    """

    document_id: int = Field(..., description="Clave primaria del documento persistido.")
    chunks_created: int = Field(..., ge=0, description="Chunks persistidos para este documento.")
    embedding_dimension: int = Field(..., description="Dimensionalidad de los vectores guardados.")
    ingestion_time_ms: int = Field(..., ge=0, description="Tiempo de pared de la ingesta.")


class SearchRequest(BaseModel):
    """Payload de ``POST /api/v1/search``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Consulta semántica en texto libre.")
    k: int = Field(default=5, ge=1, le=50, description="Número de chunks más cercanos.")


class SearchHit(BaseModel):
    """Un chunk rankeado.

    ``chunk_id`` es la clave primaria de la base de datos; el id trazable del
    corpus (``BUD-X::COMP-Y``) viaja dentro de ``metadata``.
    """

    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float = Field(..., description="Distancia coseno (menor = más similar).")
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    """Respuesta de ``POST /api/v1/search``."""

    query: str
    k: int
    search_time_ms: int = Field(..., ge=0)
    results: list[SearchHit]
```

Añade `from typing import Any` al bloque de imports del módulo.

> **`IngestStats` desaparece.** El contador de tokens y el coste ya no forman parte de la respuesta HTTP; siguen existiendo en el log estructurado de la ingesta (`embedding.ingest.end`). Dejar el modelo sin uso sería deuda muerta.

> **`extra="forbid"`.** Los modelos de entrada son un contrato estricto: un campo desconocido es un 422 que localiza el error, no un campo que se ignora en silencio.

> **`content: Budget` y no `content: dict`.** Validar el presupuesto en el borde es lo que garantiza que el chunker recibe exactamente el esquema que espera. El `422` ocurre antes de abrir la transacción.

### Paso 6 — Servicio de ingesta transaccional

En `app/embedding_pipeline/ingest_service.py`:

```python
"""Orquestación de la ingesta: chunk → embed → persist, en UNA transacción.

Compone el chunker estructural, el embedder de OpenAI y el store de pgvector.
La transacción única es el punto: la fila del documento se inserta primero
(para que los chunks puedan referenciarla tras el ``flush``), pero si la API de
embeddings falla después, el rollback elimina todo y no sobrevive ningún
``documents`` huérfano.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.embedding_pipeline.embedder import OpenAIEmbedder, estimate_cost_usd
from app.embedding_pipeline.schemas import Budget, IngestResponse
from app.embedding_pipeline.store.repository import ChunkStore
from app.embedding_pipeline.structural import JSONStructuralChunker

log = structlog.get_logger()


class DuplicateDocumentError(Exception):
    """Ya existe un documento con el mismo ``source_path``."""

    def __init__(self, document_id: int) -> None:
        super().__init__(f"Document already ingested (id={document_id})")
        self.document_id = document_id


class RagIngestService:
    """Persiste un presupuesto como documento + sus chunks embebidos."""

    def __init__(
        self,
        chunker: JSONStructuralChunker,
        embedder: OpenAIEmbedder,
        session_factory: async_sessionmaker[AsyncSession],
        store: ChunkStore,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._session_factory = session_factory
        self._store = store

    async def ingest(
        self,
        *,
        source_path: str,
        document_type: str,
        budget: Budget,
    ) -> IngestResponse:
        started = time.perf_counter()

        async with self._session_factory() as session, session.begin():
            # 1. Guardia de duplicado. Check-then-insert a nivel de aplicación:
            #    no es race-proof bajo ingestas idénticas concurrentes; a escala
            #    de formación es suficiente y es material del directo.
            existing_id = await self._store.find_document_id(session, source_path)
            if existing_id is not None:
                log.info(
                    "embedding.ingest.duplicate",
                    source_path=source_path,
                    document_id=existing_id,
                )
                raise DuplicateDocumentError(existing_id)

            # 2. Chunking estructural (un chunk por componente del presupuesto).
            chunks = self._chunker.chunk([budget])

            # 3. Embedding por lotes. El cliente de OpenAI es síncrono; se
            #    ejecuta en un hilo para no bloquear el event loop. Si falla, la
            #    transacción hace rollback: no queda documento huérfano.
            embedded = await asyncio.to_thread(self._embedder.embed_many, chunks)

            # 4. Documento + todos los chunks, aún sin commit.
            document_id = await self._store.persist_document_with_chunks(
                session,
                source_path=source_path,
                document_type=document_type,
                doc_metadata={
                    "budget_id": budget.budget_id,
                    "client_sector": budget.client_metadata.sector.value,
                    "year": budget.year,
                },
                embedded_chunks=embedded,
            )
            # 5. El commit ocurre al salir del `session.begin()`.

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        total_tokens = sum(chunk.token_count for chunk in embedded)
        response = IngestResponse(
            document_id=document_id,
            chunks_created=len(embedded),
            embedding_dimension=len(embedded[0].embedding) if embedded else 0,
            ingestion_time_ms=elapsed_ms,
        )
        log.info(
            "embedding.ingest.end",
            source_path=source_path,
            total_tokens=total_tokens,
            estimated_cost_usd=estimate_cost_usd(total_tokens),
            **response.model_dump(),
        )
        return response
```

> **Una sola transacción, de verdad.** El `session.begin()` abarca desde el chequeo de duplicado hasta el `add_all`. No hay `commit()` intermedio ni un segundo contexto de sesión. Si el embedder lanza, el `async with` revierte la fila del documento.

> **`asyncio.to_thread`.** `OpenAIEmbedder` es síncrono y usa `time.sleep` en el backoff de rate limit. Llamarlo directamente desde un handler async bloquearía el event loop durante segundos. El hilo mantiene el servicio respondiendo.

> **Tokens y coste siguen visibles.** La respuesta ya no los devuelve, pero el log `embedding.ingest.end` los conserva. La observabilidad no se pierde al cambiar el contrato.

### Paso 7 — Retriever y endpoints

#### 7.1 — `SemanticRetriever`

En `app/embedding_pipeline/retriever.py`:

```python
"""Retriever semántico sobre el store de pgvector (Sesión 08).

Embebe la query con el MISMO modelo usado en la ingesta (mezclar modelos de
embedding hace que las distancias no signifiquen nada) y rankea los chunks por
distancia coseno vía SQL. Sin índice vectorial y sin filtros de metadata: ambas
cosas se construyen en directo sobre este baseline.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import SearchHit, SearchResponse
from app.embedding_pipeline.store.repository import ChunkStore

log = structlog.get_logger()


class SemanticRetriever:
    """k-NN: embebe la query y rankea chunks por distancia coseno."""

    def __init__(
        self,
        embedder: OpenAIEmbedder,
        session_factory: async_sessionmaker[AsyncSession],
        store: ChunkStore,
    ) -> None:
        self._embedder = embedder
        self._session_factory = session_factory
        self._store = store

    async def search(self, *, query: str, k: int) -> SearchResponse:
        started = time.perf_counter()

        # Cliente de OpenAI síncrono → hilo, mismo razonamiento que en ingest.
        query_vector = await asyncio.to_thread(self._embedder.embed_one, query)

        async with self._session_factory() as session:
            matches = await self._store.search(
                session,
                query_vector=query_vector,
                k=k,
            )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        response = SearchResponse(
            query=query,
            k=k,
            search_time_ms=elapsed_ms,
            results=[
                SearchHit(
                    chunk_id=match.chunk_id,
                    document_id=match.document_id,
                    chunk_type=match.chunk_type,
                    content=match.content,
                    distance=match.distance,
                    metadata=match.metadata,
                )
                for match in matches
            ],
        )
        log.info(
            "embedding.search.end",
            query=query[:80],
            k=k,
            results=len(response.results),
            search_time_ms=elapsed_ms,
        )
        return response
```

#### 7.2 — `POST /api/v1/embeddings/ingest`

Refactoriza el handler de `app/embedding_pipeline/router.py`. El endpoint pasa a ser `async` porque espera al servicio:

```python
from fastapi.responses import JSONResponse

from app.dependencies import (
    ALL_STRATEGIES,
    build_chunkers,
    get_embedder,
    get_rag_ingest_service,
)
from app.embedding_pipeline.ingest_service import (
    DuplicateDocumentError,
    RagIngestService,
)
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    service: Annotated[RagIngestService | None, Depends(get_rag_ingest_service)],
) -> IngestResponse | JSONResponse:
    """Chunquea, embebe y persiste un presupuesto en una sola transacción."""
    if service is None:
        log.error("embedding.ingest.failed", reason="ingest_service_unavailable")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudieron persistir los embeddings.",
        )

    try:
        return await service.ingest(
            source_path=request.source_path,
            document_type=request.document_type,
            budget=request.content,
        )
    except DuplicateDocumentError as exc:
        # El contrato pide ``detail`` y ``document_id`` en el nivel superior,
        # no anidados dentro de ``detail``. Por eso se devuelve un JSONResponse
        # en vez de un HTTPException con dict.
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "detail": "Document already ingested",
                "document_id": exc.document_id,
            },
        )
    except Exception as exc:  # noqa: BLE001 — errores de embeddings/BD → 500
        log.error(
            "embedding.ingest.failed",
            reason="ingest_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudieron persistir los embeddings.",
        ) from exc
```

Elimina del router los imports que solo usaba el contrato viejo (`IngestStats`, `estimate_cost_usd`, `get_chunker` si ya no se usa en `/ingest`). **No toques** el handler `/compare` ni su lógica.

> **409 con el cuerpo exacto.** `HTTPException(detail={"document_id": 42})` produce `{"detail": {"document_id": 42}}`, que no es el contrato. `JSONResponse` permite escribir `{"detail": "...", "document_id": 42}` tal cual. El `response_model=IngestResponse` no valida esta rama porque FastAPI respeta la `Response` devuelta.

> **El `500` no filtra detalles.** El cliente recibe un mensaje genérico; el tipo de error y los primeros 300 caracteres quedan en el log. Es la misma política que en la Sesión 07.

#### 7.3 — `POST /api/v1/search`

Crea `app/routers/search.py`:

```python
"""Capa HTTP de la búsqueda semántica (Sesión 08).

Router fino: la validación vive en ``SearchRequest`` (k entre 1 y 50 → 422) y
el ranking vive en ``SemanticRetriever``. Un corpus vacío es un 200 con
``results: []``, no un error.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_semantic_retriever
from app.embedding_pipeline.retriever import SemanticRetriever
from app.embedding_pipeline.schemas import SearchRequest, SearchResponse

log = structlog.get_logger()

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    retriever: Annotated[SemanticRetriever | None, Depends(get_semantic_retriever)],
) -> SearchResponse:
    """Devuelve los ``k`` chunks más cercanos a la query por distancia coseno."""
    if retriever is None:
        log.error("embedding.search.failed", reason="retriever_unavailable")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo ejecutar la búsqueda semántica.",
        )

    try:
        return await retriever.search(query=request.query, k=request.k)
    except Exception as exc:  # noqa: BLE001 — embeddings/BD → 500
        log.error(
            "embedding.search.failed",
            reason="search_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo ejecutar la búsqueda semántica.",
        ) from exc
```

Registra el router en `app/main.py`:

```python
from app.routers import config as config_router
from app.routers import estimations, ingestion, search, sessions

# ...

app.include_router(estimations.router, prefix="/api/v1", tags=["estimations"])
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(ingestion.router, prefix="/api/v1")
app.include_router(embeddings_router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(config_router.router, prefix="/api/v1")
```

La ruta final es **`POST /api/v1/search`**. No es `/search` a secas: el repo entero vive bajo `/api/v1`.

> **Nota sobre rendimiento.** En esta fase no hay índice vectorial: Postgres hace un sequential scan completo. Para 17 documentos y 60 chunks eso es perfectamente aceptable y el endpoint responde en pocos cientos de milisegundos. No optimices la latencia todavía: observarla sin índice es justamente el punto de partida del directo.

#### 7.4 — Fábricas de dependencias

En `app/dependencies.py`, añade al bloque de imports:

```python
from app.embedding_pipeline.ingest_service import RagIngestService
from app.embedding_pipeline.retriever import SemanticRetriever
from app.embedding_pipeline.store.repository import ChunkStore
from app.persistence.database import get_async_session_factory
```

Y al final del módulo:

```python
# --- Sesión 8: persistencia pgvector + búsqueda semántica --------------------


@lru_cache
def get_chunk_store() -> ChunkStore:
    """Capa de acceso async, sin estado, sobre documents/chunks."""
    return ChunkStore()


@lru_cache
def get_rag_ingest_service() -> RagIngestService | None:
    """Orquestación chunk → embed → persist.

    ``None`` sin embeddings configurados (mismo contrato que ``get_embedder``);
    el router lo traduce a un 500 con mensaje genérico.
    """
    embedder = get_embedder()
    if embedder is None:
        return None
    return RagIngestService(
        chunker=get_chunker(),
        embedder=embedder,
        session_factory=get_async_session_factory(),
        store=get_chunk_store(),
    )


@lru_cache
def get_semantic_retriever() -> SemanticRetriever | None:
    """Contraparte de consulta del servicio de ingesta. Mismo contrato ``None``."""
    embedder = get_embedder()
    if embedder is None:
        return None
    return SemanticRetriever(
        embedder=embedder,
        session_factory=get_async_session_factory(),
        store=get_chunk_store(),
    )
```

> **No reutilizamos el embedder de la caché semántica.** `get_embedder` es la única fábrica del `OpenAIEmbedder` explícito del pipeline. La caché semántica usa `redisvl.OpenAITextVectorizer`, que es una pieza distinta.

> **Las fábricas son `lru_cache`.** Igual que el resto del repo: una instancia por worker. Los tests las limpian en `tests/conftest.py` y las sobreescriben con `app.dependency_overrides`.

Añade estas líneas al fixture autouse de `tests/conftest.py` (import y `cache_clear` en el setup y en el teardown):

```python
from app.dependencies import (
    # ...
    get_async_session_factory,
    get_chunk_store,
    get_rag_ingest_service,
    get_semantic_retriever,
)
from app.persistence.database import create_async_engine_from_settings
```

```python
create_async_engine_from_settings.cache_clear()
get_async_session_factory.cache_clear()
get_chunk_store.cache_clear()
get_rag_ingest_service.cache_clear()
get_semantic_retriever.cache_clear()
```

### Paso 8 — Script `query_examples.py` y evidencia

Crea `scripts/query_examples.py`. Sustituye el rol de `compare.py` (que medía similitud entre dos textos sueltos): ahora se ejercita la ruta real —HTTP contra `/embeddings/ingest` y `/search`— con cinco queries que atacan el corpus desde ángulos distintos.

> **`compare.py` no se borra.** Sigue siendo el sanity check de similitud de la Sesión 07 y está documentado en el README. `query_examples.py` es el CLI de retrieval de la Sesión 08: reemplaza su rol operativo, no su historia.

Requisitos del script:

1. **Idempotente.** Ingesta `data/budgets_sample.json` documento a documento. Un `409` significa "ya ingerido" y se salta; nunca duplica.
2. **Resuelve la URL base.** Respeta `ESTIMATOR_BASE_URL`; si no existe, prueba `http://localhost:8000` y `http://api:8000` (el nombre del servicio en Compose). Comprueba `GET /health` antes de seguir.
3. **Cinco queries representativas:** componente directo, reformulación semántica, dominio distinto, consulta ambigua y consulta muy específica.
4. **Top-5 legible:** `chunk_id`, `distance` con 4 decimales, `chunk_type` y los primeros 120 caracteres del `content`.
5. **Sale con código 1** si el corpus no se puede ingerir o la API no responde.

```python
#!/usr/bin/env python3
"""Smoke test de búsqueda semántica contra el corpus persistido (Sesión 08).

Reemplaza el rol de ``compare.py``: en vez de medir similitud entre dos textos
sueltos, ejercita la ruta real —HTTP contra ``POST /api/v1/embeddings/ingest``
y ``POST /api/v1/search``— con cinco queries que prueban el corpus desde
ángulos distintos (match directo, reformulación semántica, dominio ajeno,
ambigua y muy específica).

Idempotente: primero ingesta ``data/budgets_sample.json`` (un documento por
presupuesto); los documentos ya persistidos responden 409 y se saltan, así que
reejecutar el script nunca duplica datos.

Uso::

    # stack arriba primero: docker compose up -d --build api
    docker compose run --rm api python scripts/query_examples.py \
        | tee data/output/session08/output_examples.txt

    # o desde el host (con la API en localhost:8000):
    uv run python scripts/query_examples.py

La URL base se toma de ``ESTIMATOR_BASE_URL`` si existe; si no, el script
prueba ``http://localhost:8000`` y ``http://api:8000`` (alias de Compose).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "budgets_sample.json"

CANDIDATE_BASE_URLS = ("http://localhost:8000", "http://api:8000")

QUERIES: list[tuple[str, str]] = [
    (
        "Componente directo conocido (sanity check)",
        "REST API development with JWT authentication for financial sector",
    ),
    (
        "Reformulación semántica (mismo concepto, otro vocabulario)",
        "secure backend service with token-based access control for banking applications",
    ),
    (
        "Dominio distinto (no debería estar en el corpus)",
        "mobile application for restaurant reservations",
    ),
    (
        "Consulta ambigua (sin match dominante)",
        "integration with external system",
    ),
    (
        "Consulta muy específica (vocabulario técnico preciso)",
        "migration from monolith to microservices architecture using Kubernetes",
    ),
]

TOP_K = 5
CONTENT_PREVIEW_CHARS = 120


def resolve_base_url(client: httpx.Client) -> str:
    """Respeta ``ESTIMATOR_BASE_URL`` o prueba los candidatos habituales."""
    explicit = os.environ.get("ESTIMATOR_BASE_URL")
    candidates = (explicit,) if explicit else CANDIDATE_BASE_URLS
    for base_url in candidates:
        try:
            if client.get(f"{base_url}/health").status_code == 200:
                return base_url
        except httpx.TransportError:
            continue
    print(
        "ERROR: no hay API del estimador accesible. Levanta el stack "
        "(docker compose up -d --build api) o define ESTIMATOR_BASE_URL.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def ingest_corpus(client: httpx.Client, base_url: str) -> None:
    """Un documento por presupuesto; 409 = ya ingerido (idempotente)."""
    budgets = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    created, skipped = 0, 0
    for budget in budgets:
        response = client.post(
            f"{base_url}/api/v1/embeddings/ingest",
            json={
                "source_path": f"data/budgets_sample.json::{budget['budget_id']}",
                "document_type": "historical_budget",
                "content": budget,
            },
        )
        if response.status_code == 200:
            created += 1
        elif response.status_code == 409:
            skipped += 1
        else:
            print(
                f"ERROR ingiriendo {budget['budget_id']}: "
                f"{response.status_code} {response.text[:200]}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    print(f"Corpus: {len(budgets)} presupuestos — {created} ingeridos, {skipped} ya presentes.")


def run_queries(client: httpx.Client, base_url: str) -> None:
    for index, (label, query) in enumerate(QUERIES, start=1):
        response = client.post(
            f"{base_url}/api/v1/search",
            json={"query": query, "k": TOP_K},
        )
        response.raise_for_status()
        body = response.json()

        print()
        print(f"[{index}/5] {label}")
        print(f'    query: "{query}"')
        print(f"    search_time_ms: {body['search_time_ms']}")
        print(f"    {'chunk_id':>8}  {'distance':>8}  {'chunk_type':<18}  content")
        for hit in body["results"]:
            preview = " ".join(hit["content"].split())[:CONTENT_PREVIEW_CHARS]
            print(
                f"    {hit['chunk_id']:>8}  {hit['distance']:>8.4f}  "
                f"{hit['chunk_type']:<18}  {preview}"
            )


def main() -> int:
    with httpx.Client(timeout=120.0) as client:
        base_url = resolve_base_url(client)
        print(f"Estimator API: {base_url}")
        ingest_corpus(client, base_url)
        run_queries(client, base_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Ejecuta y guarda la evidencia:

```bash
mkdir -p data/output/session08
docker compose up -d --build api
docker compose run --rm api python scripts/query_examples.py \
  | tee data/output/session08/output_examples.txt
```

Y desde el host, con la API escuchando en `localhost:8000`:

```bash
uv run python scripts/query_examples.py
```

> **`container_name` no es una URL.** El script prueba `http://api:8000` porque `api` es el nombre del servicio en Compose. `estimador-cag` es el `container_name` y no resuelve por DNS dentro de la red de Compose.

> **La evidencia se versiona.** `data/output/session08/output_examples.txt` es parte del entregable: demuestra que el pipeline se ejecutó contra el corpus real y deja las distancias observadas para la conversación del directo.

> **El script no crea el directorio de salida.** El `mkdir -p` es del operador; el script solo imprime por stdout. `tee` escribe el fichero en el host.

### Paso 9 — README y tests

#### 9.1 — README

Añade una sección **"Sesión 8 — Persistencia vectorial y búsqueda semántica"** de **máximo una página**. Debe contener:

1. El contrato nuevo de los dos endpoints, con códigos `200 / 409 / 422 / 500`.
2. Los comandos para aplicar la migración y ejecutar `query_examples.py`.
3. La referencia a `data/output/session08/output_examples.txt`.
4. Las cuatro justificaciones, en este orden:
   - **(a) Dos tablas y no una.** Un presupuesto produce N chunks: es un uno-a-muchos real. Una tabla única duplicaría la metadata del documento en cada fila y perdería integridad referencial. `ON DELETE CASCADE` elimina los chunks al borrar el documento.
   - **(b) `metadata` como JSONB y no columnas.** Lo estable (tipo de documento, tipo de chunk, fechas) va en columnas tipadas; lo que el chunker enriquece (sector, tecnologías, horas) va a JSONB. El índice GIN permite consultar claves arbitrarias sin una migración por cada clave nueva. Una columna se promociona a tipada solo cuando se convierte en filtro caliente.
   - **(c) `cosine_distance` y no L2 ni inner product.** Los embeddings de OpenAI están normalizados, así que el ranking sería equivalente; usamos coseno por convención RAG y, sobre todo, para quedar alineados con la operator class `vector_cosine_ops` del índice HNSW que se añade en el directo. Un operador y una operator class distintos hacen que Postgres ignore el índice **en silencio**.
   - **(d) Sin índice vectorial todavía.** Con 17 presupuestos y 60 chunks, el sequential scan responde en pocos cientos de ms y es el baseline contra el que se mide HNSW. Añadirlo ahora ocultaría justo lo que queremos observar.

Además, actualiza la sección de la **Sesión 07** para avisar de que el contrato de `/embeddings/ingest` cambió: ya no devuelve `chunks` ni `stats`; persiste y devuelve identificadores y métricas. El `/embeddings/compare` y el Chunking Lab siguen igual.

> **Una página, no un ensayo.** El README ya es largo. La sección nueva debe ser una tabla y cuatro bullets; el detalle vive en este spec.

#### 9.2 — Tests

No hay tests de integración con una base de datos viva: la suite no tiene fixtures de Postgres y no queremos introducir infraestructura en CI. La evidencia end-to-end es `query_examples.py`. Los tests HTTP usan dobles vía `app.dependency_overrides`.

**`tests/test_embedding_endpoint.py`** — reescríbelo para el contrato nuevo:

```python
class FakeIngestService:
    """Doble del servicio de ingesta: no toca Postgres ni la API de embeddings."""

    def __init__(
        self,
        *,
        response: IngestResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or IngestResponse(
            document_id=42,
            chunks_created=3,
            embedding_dimension=1536,
            ingestion_time_ms=12,
        )
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def ingest(
        self,
        *,
        source_path: str,
        document_type: str,
        budget: Budget,
    ) -> IngestResponse:
        self.calls.append(
            {
                "source_path": source_path,
                "document_type": document_type,
                "budget": budget,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response
```

Casos:

- `200` con `{"source_path", "document_type", "content"}` válido; comprueba `document_id`, `chunks_created`, `embedding_dimension`, `ingestion_time_ms` y que el servicio recibió el `source_path` y el `document_type`.
- `409` con `DuplicateDocumentError(42)`; comprueba el cuerpo exacto `{"detail": "Document already ingested", "document_id": 42}`.
- `422` con un sector inválido y con un campo desconocido en el presupuesto.
- `422` con `source_path` vacío o `document_type` de más de 50 caracteres.
- `500` cuando `get_rag_ingest_service` devuelve `None`.

**`tests/test_search_endpoint.py`** — nuevo:

```python
class FakeRetriever:
    async def search(self, *, query: str, k: int) -> SearchResponse:
        return SearchResponse(
            query=query,
            k=k,
            search_time_ms=7,
            results=[
                SearchHit(
                    chunk_id=1,
                    document_id=2,
                    chunk_type="budget_component",
                    content="Backend service implementation with JWT...",
                    distance=0.231,
                    metadata={"budget_id": "BUD-2024-001"},
                )
            ],
        )
```

Casos:

- `200` con `{"query": "...", "k": 5}` y forma completa de cada `SearchHit`.
- `422` con `k=0`, `k=51` o `query` vacía.
- `500` cuando `get_semantic_retriever` devuelve `None`.

**`tests/test_embedding_chunker.py`** — deja de usar `IngestRequest(budgets=payload)`:

```python
def _budgets() -> list[Budget]:
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    return [Budget.model_validate(item) for item in payload]
```

Y cambia los dos tests de validación para usar `Budget.model_validate(payload[0])`. El resto de aserciones (un chunk por componente, metadata, cabecera de contexto, `token_count`) no cambia.

**`tests/test_project_structure.py`** — añade a `REQUIRED_PATHS`:

```python
REQUIRED_PATHS += (
    "app/embedding_pipeline/store/__init__.py",
    "app/embedding_pipeline/store/models.py",
    "app/embedding_pipeline/store/repository.py",
    "app/embedding_pipeline/ingest_service.py",
    "app/embedding_pipeline/retriever.py",
    "app/routers/search.py",
    "alembic/versions/0002_session8_pgvector.py",
    "scripts/query_examples.py",
    "data/output/session08/output_examples.txt",
)
```

Y a `test_app_exposes_expected_entrypoints`:

```python
assert "/api/v1/search" in paths
```

> **`client` fixture y handlers async.** `TestClient` ejecuta la app en un portal async; un handler `async def` con un doble `async def` funciona sin `pytest-asyncio` adicional. No hace falta levantar Postgres.

> **No se testea el SQL real.** Sin una base viva no hay forma honesta de validar `cosine_distance`. Eso se cubre con `query_examples.py` y se discutirá en el directo.

### Entregable

Una rama nueva en el repositorio: **`feature/pre-session-08`**. Debe contener:

1. `pyproject.toml` y `uv.lock` con `asyncpg`, `pgvector` y `greenlet`.
2. `app/persistence/database.py` con el motor async y la factoría de sesiones.
3. `app/embedding_pipeline/store/` completo (`__init__.py`, `models.py`, `repository.py`).
4. `app/embedding_pipeline/ingest_service.py` y `app/embedding_pipeline/retriever.py`.
5. `app/embedding_pipeline/schemas.py` con los contratos nuevos y sin `IngestStats`.
6. `app/embedding_pipeline/router.py` con el ingest persistente y el `/compare` intacto.
7. `app/routers/search.py` y el registro en `app/main.py`.
8. `app/dependencies.py` y `tests/conftest.py` con las fábricas y sus `cache_clear`.
9. `alembic/versions/0002_session8_pgvector.py` y `alembic/env.py` actualizado.
10. `scripts/query_examples.py` ejecutable.
11. `data/output/session08/output_examples.txt` con la salida real.
12. README con la sección nueva y la nota de cambio de contrato en la Sesión 07.
13. Tests actualizados (`test_embedding_endpoint.py`, `test_embedding_chunker.py`, `test_project_structure.py`) y nuevo `test_search_endpoint.py`.

No es obligatorio añadir tests de integración con Postgres. Si decides escribirlos, perfecto, pero no son requisito.

## Criterios de "hecho"

El ejercicio está completo cuando:

- [ ] `POST /api/v1/embeddings/ingest` acepta `{source_path, document_type, content}` y devuelve `{document_id, chunks_created, embedding_dimension, ingestion_time_ms}`.
- [ ] Un `source_path` repetido devuelve **409** con `{"detail": "Document already ingested", "document_id": N}`.
- [ ] Un presupuesto inválido devuelve **422** antes de abrir transacción o llamar a la API de embeddings.
- [ ] El documento y sus chunks se escriben en **una sola transacción**; un fallo del embedder no deja `documents` huérfanos.
- [ ] `POST /api/v1/search` devuelve los `k` chunks más cercanos por distancia coseno, con `chunk_id`, `document_id`, `chunk_type`, `content`, `distance` y `metadata`.
- [ ] `alembic upgrade head` crea la extensión `vector`, las tablas `documents` y `chunks`, y los índices no vectoriales.
- [ ] `scripts/query_examples.py` ingesta el corpus (17 documentos / 60 chunks) de forma idempotente y ejecuta las cinco queries.
- [ ] `data/output/session08/output_examples.txt` contiene la salida real del script.
- [ ] El README justifica (a) dos tablas, (b) JSONB, (c) coseno y (d) ausencia de índice vectorial.
- [ ] `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` y `uv run mypy app frontend` pasan.

## Checklist de verificación

- [ ] `asyncpg`, `pgvector` y `greenlet` están en `pyproject.toml`; `sqlalchemy` y `alembic` no se duplican.
- [ ] `get_async_session_factory` deriva `+asyncpg` de `settings.database_url`; no hay una segunda variable de entorno.
- [ ] `SessionLocal`, `get_session()` y los repositorios de la Sesión 06 no se han tocado.
- [ ] `DocumentRow` y `ChunkRow` usan `Base` de `app.persistence.models`; no hay una segunda `Base`.
- [ ] `metadata_` está mapeado a la columna física `"metadata"` en ambas tablas.
- [ ] `embedding` es `Vector(1536)` y `nullable=True`.
- [ ] `ON DELETE CASCADE` está en la FK de `chunks.document_id`.
- [ ] La migración es `0002_session8_pgvector` con `down_revision = "0001_session6_initial"`.
- [ ] `alembic/env.py` importa `app.embedding_pipeline.store.models` y registra `ischema_names["vector"] = Vector`.
- [ ] No existe ningún índice HNSW/IVFFlat ni filtro de metadata en SQL.
- [ ] `ChunkStore` no hace `commit()`; el `session.begin()` del servicio es la única frontera transaccional.
- [ ] El embedder síncrono se invoca con `asyncio.to_thread` tanto en ingest como en search.
- [ ] El ingest usa el mismo `EMBEDDING_MODEL` que la búsqueda; no hay modelos mezclados.
- [ ] El `409` devuelve `detail` y `document_id` en el nivel superior (JSONResponse, no HTTPException anidada).
- [ ] `IngestStats` ha desaparecido; el coste y los tokens siguen en el log `embedding.ingest.end`.
- [ ] `POST /api/v1/search` aparece en `/docs` y en `app.openapi()["paths"]`.
- [ ] `scripts/query_examples.py` prueba `http://localhost:8000` y `http://api:8000`, y respeta `ESTIMATOR_BASE_URL`.
- [ ] `data/output/session08/output_examples.txt` está versionado y no lo excluye `.gitignore`.
- [ ] `tests/conftest.py` limpia las cachés nuevas.
- [ ] `tests/test_embedding_chunker.py` ya no construye `IngestRequest(budgets=...)`.
- [ ] `tests/test_project_structure.py` incluye las rutas nuevas y `/api/v1/search`.

## Documentación de referencia

- pgvector — extensión y operadores de distancia: https://github.com/pgvector/pgvector
- pgvector-python — tipos y operadores de SQLAlchemy: https://github.com/pgvector/pgvector-python
- SQLAlchemy 2.0 — `create_async_engine`, `async_sessionmaker` y `AsyncSession`: https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html
- asyncpg — driver async de PostgreSQL: https://magicstack.github.io/asyncpg/current/
- Alembic — operaciones, `create_table`, `create_index` y `op.execute`: https://alembic.sqlalchemy.org/en/latest/ops.html
- FastAPI — `JSONResponse`, `Depends` y `response_model`: https://fastapi.tiangolo.com/advanced/response-directly/
- PostgreSQL — JSONB y GIN: https://www.postgresql.org/docs/16/datatype-json.html
- PostgreSQL — `EXPLAIN` y planes de ejecución (material del directo): https://www.postgresql.org/docs/16/using-explain.html

## Nota

**Por qué dos tablas y no una.** Un presupuesto produce N chunks; eso es un uno-a-muchos, no una tabla ancha. Si se mete todo en `chunks`, la metadata del documento (`source_path`, `document_type`, `ingested_at`) se repite en cada fila y nada impide que dos chunks del mismo presupuesto discrepen sobre su procedencia. Con `documents` + `chunks` y `ON DELETE CASCADE`, borrar un presupuesto borra sus chunks automáticamente y la procedencia vive en un único sitio. El coste es un `JOIN` cuando se necesita; el beneficio es integridad referencial.

**Por qué JSONB y no columnas.** Hay dos clases de metadata. La estable —tipo de documento, tipo de chunk, fechas— se consulta, se indexa y se tipa: columna. La variable —sector, tecnologías, horas, lo que el chunker decida añadir mañana— no justifica una migración por cada clave nueva. JSONB + GIN da consultas por claves arbitrarias (`metadata->>'sector'`) sin tocar el esquema. La regla de promoción es simple: cuando un campo se convierte en filtro caliente, se promociona a columna tipada. En esta sesión no hay filtros; el GIN es la puerta que el directo abre.

**Por qué `cosine_distance` y no L2 ni inner product.** Con embeddings normalizados de OpenAI, los tres rankings son equivalentes. La elección no es matemática, es operativa: coseno es la convención de la literatura RAG y, sobre todo, es la operator class (`vector_cosine_ops`) con la que se construye el índice HNSW del directo. Si la query usa `<=>` y el índice está en `vector_l2_ops`, Postgres **ignora el índice en silencio** y vuelve al sequential scan. Alinear operador y operator class desde el principio evita un bug de rendimiento invisible.

**Por qué no hay índice vectorial todavía.** Con 17 presupuestos y 60 chunks, un sequential scan responde en pocos cientos de milisegundos. Añadir HNSW ahora escondería el baseline: no sabríamos cuánto mejora ni cuánto cuesta mantenerlo. El ejercicio mide el `EXPLAIN ANALYZE` sin índice; el directo añade el índice y compara. La disciplina de no adelantarse es parte del valor.

**Qué es nuestro (y por qué).** Frente a la implementación canónica del enunciado, este repo toma decisiones deliberadas: (1) el store vive en `app/embedding_pipeline/store/`, no en un `generation/rag` importado de otro proyecto; (2) no se añade un servicio `postgres` porque `estimator-postgres` ya existe con pgvector desde la Sesión 06; (3) el stack async convive con el sync de la Sesión 06 en lugar de reescribirlo; (4) el repositorio devuelve un `ChunkMatch` tipado, no un `Row` de SQLAlchemy; (5) el embedder síncrono se ejecuta con `asyncio.to_thread` para no bloquear el event loop; (6) el `409` usa `JSONResponse` para cumplir el cuerpo exacto del contrato; (7) la evidencia se guarda en `data/output/session08/output_examples.txt`, dentro de la estructura de datos del repo; y (8) los tokens y el coste siguen en el log aunque la respuesta HTTP ya no los devuelva.

**El contrato viejo se va.** El `/embeddings/ingest` de la Sesión 07 devolvía vectores por HTTP. Mantenerlo en paralelo obligaría a dos caminos, dos respuestas y dos formas de razonar sobre la misma operación. Se reemplaza. El `/embeddings/compare` y el Chunking Lab no dependen de él y siguen funcionando. La sección de la Sesión 07 del README se actualiza para que nadie busque un contrato que ya no existe.

**`Vector(1536)` no es configuración.** Es la dimensionalidad de `text-embedding-3-small`. Cambiar de modelo o de dimensión (Matryoshka, `text-embedding-3-large`, un endpoint custom) implica re-embedear el corpus entero y migrar la columna. Por eso no se lee de una variable de entorno: fingir que es dinámico sería mentir sobre el coste de cambiarlo. La comparativa Matryoshka del endpoint `/embeddings/compare` no se persiste.

**Fuera de alcance.** Índices vectoriales (HNSW/IVFFlat), filtros por metadata, búsqueda híbrida, tuning de Postgres, métricas de retrieval, re-embedding masivo y cualquier cambio en la UI o en `cliente-web`. Todo eso es del directo. Este ejercicio **persiste y busca**; no optimiza.
