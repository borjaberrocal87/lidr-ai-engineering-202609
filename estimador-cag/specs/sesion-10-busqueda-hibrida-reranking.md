# Spec — Proyecto 4: Búsqueda híbrida y reranking (Sesión 10)

## Objetivo

El pipeline RAG de la Sesión 09 ya funciona de extremo a extremo: destila un brief, embebe la
consulta, recupera presupuestos por similitud vectorial y genera una estimación con ese contexto.
El problema es que **“similar” no siempre significa “relevante”**: una consulta sobre una
plataforma de e-commerce puede recuperar el presupuesto de una app de pagos porque ambos viven en
la misma región del espacio vectorial. Cercano, pero inútil para estimar.

Esta sesión ataca ese problema con dos técnicas y, sobre todo, **mide si compensan**:

1. **Búsqueda híbrida** — una rama léxica (full-text search de PostgreSQL) se fusiona con la rama
   vectorial existente mediante **Reciprocal Rank Fusion (RRF)**.
2. **Reranking** — un **cross-encoder** reordena los candidatos que la recuperación amplia trajo
   baratos, siguiendo el patrón *recall-then-rerank*: recuperar ancho (top-50) y quedarse con los
   mejores (top-5).

El objetivo no es solo que el retrieval mejore: es **demostrar con números cuánto mejora y a qué
coste**. Para eso se construye un golden set de cinco consultas anotadas a mano y se comparan cuatro
configuraciones (vectorial/híbrida × reranking sí/no) con precisión@5 y latencia.

Al terminar este ejercicio tendrás:

- Una migración Alembic con una columna `tsvector` generada y su índice GIN.
- Una rama léxica en `ChunkStore` y una función pura de fusión RRF.
- Un pipeline de retrieval con las cuatro configuraciones, activables sin tocar código.
- El wrapper de cross-encoder integrado (no implementado desde cero).
- Un golden set de cinco consultas y un script que produce la tabla comparativa.
- Un documento `docs/sesion-10-busqueda-hibrida-reranking.md` con metodología, resultados y
  conclusiones.

> **Alcance estricto.** Esta sesión cubre **solo** búsqueda híbrida y reranking. No se implementan
> expansión de consultas, routing multi-índice ni filtros de metadata nuevos. Los filtros
> estructurales de la Sesión 09 (sector, año, tipo de chunk) se **reutilizan tal cual** en ambas
> ramas para que las cuatro configuraciones comparen el mismo espacio de candidatos.

## Reconciliación con el repo

El punto de partida es `main` con la Sesión 09 cerrada: pipeline RAG grounded, endpoints de
retrieval por etapas, corpus de presupuestos y corpus task-level ingeridos, tests y cliente web.
**Nada de eso se reescribe.** En particular:

- La arquitectura por capas manda. El código de retrieval vive en `app/generation/rag/`, el
  transporte HTTP en `app/api/`, la configuración en `app/config.py` y el *composition root* en
  `app/dependencies.py`. No se crean carpetas sueltas en la raíz.
- El contrato de retrieval de la Sesión 09 se llama `RetrievalOutcome` y su chunk tipado
  `RetrievedChunk`. **No se renombran** a `RetrievalResult` ni se rompen los endpoints del wizard.
- El store devuelve `ChunkMatch`, no `Row` crudos de SQLAlchemy. La nueva rama léxica respeta ese
  contrato.
- Los errores RAG viven en `app/generation/rag/errors.py`; el pipeline usará `RagRetrievalError`,
  no un `RetrievalError` importado de otro proyecto.
- El servicio de Compose se llama `api`, no `ai-service` ni `estimator`. Los comandos de
  verificación se adaptan.
- El corpus canónico (`data/budgets_sample.json`, `data/task_corpus.json`) está en **inglés**,
  aunque el enunciado diga que está en español. La configuración de text search será `english` y
  se documenta el porqué. Si algún día se ingiere un corpus español, el cambio es explícito y
  afecta a dos sitios.
- El repo ya tiene `data/task_corpus.json` con `chunk_type="historical_task"`. El golden set de esta
  sesión se evalúa **solo sobre `budget_component`**, para que la métrica sea reproducible aunque
  el corpus task-level también esté ingerido.
- El repo **no tiene todavía** `app/generation/rag/retrieval/`. El wrapper del cross-encoder y el
  script de verificación se entregan como material en el Paso 0; el ejercicio consiste en
  **integrarlos**, no en reimplementarlos.

### Piezas que se reutilizan

- `app/generation/rag/store/repository.py` — `ChunkStore`, `ChunkMatch` y la búsqueda vectorial
  filtrada (`search_with_filters`).
- `app/generation/rag/store/models.py` — `DocumentRow`, `ChunkRow` y la `Base` de la Sesión 06.
- `app/generation/rag/schemas.py` — `Chunk`, `EmbeddedChunk`, `RetrievalQuery`, `RetrievedChunk`,
  `RetrievalOutcome` y los contratos del wizard.
- `app/generation/rag/retriever.py` — `SemanticRetriever`, que ya embebe la query y expone
  `retrieve_chunks`.
- `app/generation/rag/grounded_estimator.py` — el conductor del pipeline transcripción →
  estimación.
- `app/generation/rag/errors.py` — `RagRetrievalError`.
- `app/generation/rag/embedding/embedder.py` — el `OpenAIEmbedder` real del repo.
- `app/config.py` — settings con `@lru_cache`.
- `app/foundation/llm/runtime_config.py` — `RuntimeModelConfig`, el patrón Redis de overrides en
  caliente. Aquí se añade `RuntimeRetrievalConfig`.
- `app/dependencies.py` — fábricas `lru_cache` (`get_chunk_store`, `get_semantic_retriever`,
  `get_runtime_config`…).
- `app/api/retrieval.py` — `POST /api/v1/retrieval/search`.
- `app/api/config.py` — `GET/PUT /api/v1/config/models`. Aquí se añaden los toggles de retrieval.
- `alembic/` — migraciones `0001_session6_initial` y `0002_session8_pgvector`. La nueva es
  `0003_session10_fts`.
- `scripts/rag_ops/common.py` — `Stopwatch`, `require_embedder` y utilidades de benchmark.
- `tests/test_retrieval_filtered.py`, `tests/test_search_halfvec.py` y `tests/test_rag_*.py` — el
  estilo de tests con dobles, sin Postgres ni red.

## Requisitos para el ejercicio

- Proyecto de la Sesión 09 en verde:

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy app frontend
  ```

- Stack levantado desde `estimador-cag/`:

  ```bash
  docker compose up -d --build
  curl -s http://localhost:8000/health | jq
  ```

  El servicio `api` aplica `alembic upgrade head` al arrancar. Si `/health` no responde, no
  empieces el ejercicio.

- Corpus base ingerido de forma idempotente:

  ```bash
  docker compose run --rm api python scripts/query_examples.py
  ```

  Debes tener los **17 presupuestos / 60 chunks** de `data/budgets_sample.json`. Si además está
  ingerido `data/task_corpus.json`, no pasa nada: el golden set filtra por
  `chunk_type="budget_component"`.

- Clave de embeddings en `.env` (`OPEN_AI_KEY`) o un endpoint OpenAI-compatible
  (`EMBEDDING_BASE_URL` + `EMBEDDING_API_KEY`). El modelo debe ser el mismo de la ingesta
  (`text-embedding-3-small`).

- Dependencia nueva del reranker:

  ```bash
  uv add sentence-transformers
  ```

  `sentence-transformers` arrastra `torch`. La imagen de runtime crece y el primer uso descarga los
  pesos del modelo. Es el coste real de tener un cross-encoder local; se mide en la sesión.

- `uv.lock` actualizado y commiteado: el `Dockerfile` construye con `uv sync --frozen`.

- Docker y Docker Compose funcionando. El modelo se descarga dentro del contenedor, así que la
  primera ejecución necesita red.

## Lo que entra y lo que no entra

**Entra:**

- Migración Alembic con `content_tsv` (`tsvector` generado) e índice GIN.
- Búsqueda léxica en `ChunkStore` con `websearch_to_tsquery` (OR de términos) y `ts_rank_cd`.
- Fusión RRF pura y testeable.
- Pipeline `recall-then-rerank` con `search_mode` (`vector`/`hybrid`) y `rerank` (on/off).
- Integración del wrapper de cross-encoder entregado.
- Overrides de retrieval en runtime (Redis) y endpoints `GET/PUT /api/v1/config/retrieval`.
- Golden set de cinco consultas y script de evaluación con precisión@5 y latencia.
- Tests unitarios de reranker, fusión y pipeline.
- Documento de resultados y conclusiones.

**No entra:**

- Expansión de consultas, HyDE, traducción de queries o reformulación nueva. La comprensión de la
  Sesión 09 se queda como está.
- Routing multi-índice o búsqueda en varios stores.
- Filtros de metadata nuevos (tecnología, escala, país…). Los filtros de sector/año/tipo ya existen
  y se reutilizan.
- Cambios en la UI, en `cliente-web` o en el wizard de estimación.
- Fine-tuning del cross-encoder, comparativa de varios modelos de reranking o embeddings.
- Métricas formales más allá de precisión@5 y latencia (NDCG, MRR, recall@k completo).
- Cambios en el corpus o re-ingesta masiva. La migración calcula `content_tsv` sobre las filas
  existentes.

## Datos para el ejercicio

### Corpus y configuración de text search

- `data/budgets_sample.json`: **17 presupuestos**, **60 componentes**, cuatro sectores
  (`finance`, `ecommerce`, `healthcare`, `industrial`).
- `data/task_corpus.json`: corpus task-level de la Sesión 09 (`chunk_type="historical_task"`).
- **Idioma real del corpus: inglés.** El enunciado dice español, pero los resúmenes y descripciones
  del dataset están en inglés (“Mobile banking API with OAuth 2.0…”, “Headless e-commerce
  storefront…”). Por eso la configuración de text search es `english`: su stemming y stop-words son
  los que aportan recall léxico sobre estos datos.
- Si se ingiere un corpus en español, hay que cambiar `english` por `spanish` en **dos sitios**:
  la migración `0003_session10_fts` y la llamada a `websearch_to_tsquery` en `ChunkStore`. La columna
  generada no se puede cambiar sin recrearla. Esta duplicación es deliberada y está comentada en el
  código.

### Golden set de referencia

El golden set vive en `evals/golden_retrieval.json`. Cinco consultas representativas del dominio,
anotadas a mano contra los presupuestos del corpus base. La relevancia se anota **por presupuesto
padre** (`budget_id`), no por componente: a esta escala es más estable y alinea la métrica con cómo
se estima (un presupuesto completo, no un chunk suelto).

```json
{
  "description": "Session 10 golden set for retrieval evaluation. Five representative project descriptions to estimate, each hand-annotated with the historical budgets that are genuinely relevant. Graded over the BASE corpus (data/budgets_sample.json, chunk_type='budget_component') so the score is reproducible even if the task corpus is also ingested. A retrieved chunk counts as relevant iff its parent budget_id is in relevant_budget_ids. The distractors are deliberately cross-sector 'similar but irrelevant' budgets (e.g. e-commerce vendor payouts vs. a fintech payments gateway) — the exact failure mode this exercise attacks.",
  "corpus": "data/budgets_sample.json",
  "chunk_types": ["budget_component"],
  "k": 5,
  "queries": [
    {
      "id": "Q1",
      "query": "Mobile banking application with OAuth2 authentication, PSD2 open banking compliance, strong customer authentication and a transaction ledger.",
      "relevant_budget_ids": ["BUD-2024-001", "BUD-2024-003"],
      "note": "Fintech auth + payments. Trap: BUD-2024-006 (e-commerce marketplace vendor payouts) is 'payments-like' but irrelevant for a banking auth project."
    },
    {
      "id": "Q2",
      "query": "Headless e-commerce storefront with a product catalog, shopping cart checkout and personalized product recommendations.",
      "relevant_budget_ids": ["BUD-2024-005", "BUD-2024-006", "BUD-2024-007", "BUD-2024-017"],
      "note": "E-commerce storefront/catalog/checkout. Trap: finance payment budgets (BUD-2024-003) and healthcare scheduling are close in topic words but wrong domain."
    },
    {
      "id": "Q3",
      "query": "Telemedicine platform for patient appointment scheduling and video consultations with HL7/FHIR integration and clinical records.",
      "relevant_budget_ids": ["BUD-2024-009", "BUD-2024-010"],
      "note": "Healthcare telemedicine. Trap: BUD-2024-011 (pharmacy inventory) and BUD-2024-012 (clinical trials) share the healthcare sector but cover different functionality."
    },
    {
      "id": "Q4",
      "query": "Industrial IoT platform ingesting sensor telemetry for predictive maintenance and real-time equipment monitoring in a factory.",
      "relevant_budget_ids": ["BUD-2024-013", "BUD-2024-015"],
      "note": "Industrial telemetry/monitoring. Trap: BUD-2024-014 (warehouse AGV coordination) is industrial but logistics, not telemetry-driven maintenance."
    },
    {
      "id": "Q5",
      "query": "Real-time payments gateway with fraud detection, transaction reconciliation and a double-entry ledger.",
      "relevant_budget_ids": ["BUD-2024-003", "BUD-2024-001"],
      "note": "Fintech payments. Trap: BUD-2024-006 (e-commerce vendor payouts) and BUD-2024-007 (grocery delivery payments) are lexically near but wrong domain."
    }
  ]
}
```

> **No copies el golden set sin mirarlo.** Es un set de referencia construido sobre este corpus.
> Léelo, comprueba que estás de acuerdo con las anotaciones y ajústalo si tu criterio es distinto.
> Lo que se evalúa es que sepas justificar la relevancia, no que coincidas al 100 % con una tabla.

### Las cuatro configuraciones

| Configuración | Búsqueda | Reranking |
|---|---|---|
| A | Vectorial | No |
| B | Híbrida | No |
| C | Vectorial | Sí |
| D | Híbrida | Sí |

A es el baseline de la Sesión 09. B añade la rama léxica y RRF. C añade el cross-encoder sobre el
recall vectorial. D combina ambas técnicas.

## ✍ Ejercicio

### Paso 0 — Material entregado y gate de verificación

Crea el paquete `app/generation/rag/retrieval/` con `__init__.py`, el wrapper del cross-encoder y el
script de verificación. **El wrapper se entrega hecho**: no tienes que implementar el reranker, solo
integrarlo. Si prefieres comparar, la solución de referencia está en la rama `session_10` del repo
del programa; aquí se reproduce adaptada a nuestras rutas y a nuestros settings.

`app/generation/rag/retrieval/__init__.py`:

```python
"""Session 10 — hybrid search + cross-encoder reranking.

Two relevance techniques layered on top of the Session 9 vector retriever:

* ``fusion`` — Reciprocal Rank Fusion of the vector and lexical rankings.
* ``reranker`` — a cross-encoder that rescues precision after a wide recall.
* ``pipeline`` — the ``retrieve()`` entrypoint that composes both behind the
  ``search_mode`` (vector/hybrid) and ``rerank`` (on/off) toggles.

The package depends only on ``config``, ``foundation``/``domain`` contracts and
sibling modules under ``generation/rag`` — never on ``api`` or ``dependencies``.
"""
```

`app/generation/rag/retrieval/reranker.py`:

```python
"""Cross-encoder reranker wrapper (Session 10).

Bi-encoders (the embedding retriever) encode the query and each document
SEPARATELY, so "relevance" is the cosine of two vectors computed without either
side seeing the other — fast, but it misses fine-grained relevance. A
cross-encoder feeds the ``(query, document)`` pair through the model TOGETHER
and outputs a single relevance score, attending across both texts. Much more
accurate, much too slow to run over the whole corpus — hence recall-then-rerank:
the bi-encoder/lexical branches recall a wide candidate set cheaply, the
cross-encoder rescores only those.

This module is the wrapper (model loading + pair scoring). It does NOT decide the
recall width or the final cut — that orchestration lives in ``pipeline.retrieve``.

The model is multilingual (``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``) so it
copes with the corpus regardless of the Spanish/English question, and is small
enough to run on CPU at teaching latencies. It loads LAZILY on first use:
importing this module must stay cheap (tests, app startup) and must not require
torch weights on disk until something actually reranks.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import structlog

from app.config import get_settings

log = structlog.get_logger()


class CrossEncoderReranker:
    """Lazily-loaded cross-encoder that rescores ``(query, document)`` pairs.

    One instance owns one loaded model. Loading is guarded by a lock so the
    first concurrent rerank does not trigger two parallel downloads/loads.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any = None  # sentence_transformers.CrossEncoder, loaded on demand
        self._load_lock = threading.Lock()

    @classmethod
    def from_settings(cls) -> CrossEncoderReranker:
        return cls(get_settings().reranker_model)

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self) -> None:
        """Force-load the model now (used by the verify script and warmup)."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:  # another thread won the race
                return self._model
            # Imported here, not at module top: keeps the import graph (and app
            # startup / unit tests) free of torch until a rerank truly happens.
            from sentence_transformers import CrossEncoder

            started = time.perf_counter()
            self._model = CrossEncoder(self._model_name)
            log.info(
                "reranker.loaded",
                model=self._model_name,
                load_ms=int((time.perf_counter() - started) * 1000),
            )
            return self._model

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Relevance score for ``query`` against each document (higher = better).

        One forward pass over all ``(query, document)`` pairs (sentence-transformers
        batches internally). Returns a score per document, in input order.
        """
        if not documents:
            return []
        model = self._ensure_loaded()
        pairs = [(query, document) for document in documents]
        started = time.perf_counter()
        scores = model.predict(pairs)
        log.info(
            "reranker.scored",
            model=self._model_name,
            pairs=len(pairs),
            score_ms=int((time.perf_counter() - started) * 1000),
        )
        return [float(score) for score in scores]

    def rerank(
        self,
        query: str,
        candidates: list[Any],
        *,
        top_n: int,
        text_of: Callable[[Any], str] = lambda candidate: candidate.content,
    ) -> list[Any]:
        """Reorder ``candidates`` by cross-encoder score, keep the top ``top_n``.

        ``text_of`` extracts the document text from a candidate (default: a
        ``.content`` attribute) so this works for ORM rows, schema objects or
        plain dicts without coupling the reranker to any one shape. Order is
        stable for equal scores (Python's sort is stable and the candidates keep
        their recall order).
        """
        if not candidates:
            return []
        scores = self.score(query, [text_of(candidate) for candidate in candidates])
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [candidate for candidate, _score in ranked[:top_n]]
```

`app/generation/rag/retrieval/verify_reranker.py`:

```python
"""Pre-flight check: does the cross-encoder download, load and score?

Run BEFORE touching any retrieval code (the exercise's gate)::

    docker compose exec api python -m app.generation.rag.retrieval.verify_reranker

It loads the configured model and scores a tiny sanity pair where one document
is obviously more relevant than the other, asserting the model ranks them
correctly. Exit code 0 = ready; non-zero (with a readable reason) = stop and fix
the environment before continuing.
"""

from __future__ import annotations

import sys

from app.generation.rag.retrieval.reranker import CrossEncoderReranker

# A sanity pair: the first document answers the query, the second does not. Any
# working relevance model must score doc-0 above doc-1.
_QUERY = "e-commerce checkout and shopping cart platform"
_DOCUMENTS = [
    "Online store checkout flow with shopping cart, payment and order management.",
    "Hospital patient appointment scheduling and telemedicine video consultations.",
]


def main() -> int:
    reranker = CrossEncoderReranker.from_settings()
    print(f"Loading reranker model: {reranker.model_name} ...")
    try:
        scores = reranker.score(_QUERY, _DOCUMENTS)
    except Exception as exc:  # noqa: BLE001 — surface any load/score failure.
        print(f"FAILED to load or run the reranker: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Query:          {_QUERY!r}")
    print(f"Relevant doc:   score = {scores[0]:.4f}")
    print(f"Irrelevant doc: score = {scores[1]:.4f}")

    if scores[0] <= scores[1]:
        print(
            "WARNING: the relevant document did not outscore the irrelevant one. "
            "The model loaded, but its ranking looks off — check the model name.",
            file=sys.stderr,
        )
        return 2

    print("OK: reranker loaded and ranked the relevant document first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

#### 0.1 — Settings mínimos

El wrapper lee `settings.reranker_model`, así que añade **antes del gate** los campos de la Sesión 10
a `app/config.py`, junto a los de la Sesión 09:

```python
    # Session 10 — hybrid search + cross-encoder reranking.
    # Default retrieval mode: "vector" reproduces the Session 9 baseline;
    # "hybrid" fuses dense + lexical with RRF. Overridable per request and at
    # runtime.
    retrieval_search_mode: Literal["vector", "hybrid"] = "vector"
    # Whether the cross-encoder reranks by default. Off keeps the baseline
    # cheap; recall-then-rerank turns on per request / runtime.
    reranker_enabled: bool = False
    # Multilingual cross-encoder (ES+EN), small enough for CPU at teaching
    # latency.
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # Recall width before reranking/fusion: retrieve this many candidates
    # cheaply, then the cross-encoder rescores them down to rerank_top_n.
    retrieval_recall_top_k: int = 50
    rerank_top_n: int = 5
    # RRF smoothing constant (Cormack et al. default). Larger = a document must
    # rank well in BOTH branches to win; smaller = a single #1 can dominate.
    rrf_k: int = 60
```

Añade también los valores a `.env.example`:

```dotenv
# Sesión 10 — búsqueda híbrida (vector + léxica) y reranking.
RETRIEVAL_SEARCH_MODE=vector
RERANKER_ENABLED=false
RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
RETRIEVAL_RECALL_TOP_K=50
RERANK_TOP_N=5
RRF_K=60
```

Añade la dependencia y reconstruye la imagen:

```bash
uv add sentence-transformers
docker compose build api
docker compose up -d
```

Ejecuta el gate:

```bash
docker compose exec api python -m app.generation.rag.retrieval.verify_reranker
```

Debe terminar con `OK: reranker loaded and ranked the relevant document first.` y código de salida
0. **Si este paso falla, no sigas.** Revisa la sección de troubleshooting al final de esta spec.

> La primera ejecución descarga los pesos del modelo en la caché de Hugging Face del usuario `app`
> dentro del contenedor. Es normal que tarde. Las siguientes cargas son locales.

### Paso 1 — Columna full-text e índice GIN

Crea `alembic/versions/0003_session10_fts.py`. El `down_revision` apunta a
`0002_session8_pgvector`:

```python
"""Sesión 10 — columna full-text (tsvector) e índice GIN sobre chunks.

Revision ID: 0003_session10_fts
Revises: 0002_session8_pgvector
Create Date: 2026-11-06 00:00:00

Añade `content_tsv`, una columna generada STORED a partir de `content`, y su
índice GIN. Es la base de la rama léxica de la búsqueda híbrida: la rama
vectorial sigue usando `embedding` y ambas se fusionan con RRF.

Configuración de text search:
- El enunciado del ejercicio dice que el corpus está en español, pero el corpus
  canónico de este repo (`data/budgets_sample.json`, `data/task_corpus.json`)
  está en inglés. Usamos `english` porque su stemming y stop-words son los que
  aportan recall léxico real sobre estos datos.
- Si en el futuro se ingiere un corpus en español, hay que cambiar `english`
  por `spanish` en DOS sitios: esta migración y `websearch_to_tsquery` del
  repositorio. La columna generada no se puede cambiar sin recrearla.

La columna es GENERATED ALWAYS ... STORED: Postgres la recalcula al insertar o
actualizar `content`, así que no hay trigger ni índice que pueda desincronizarse.
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
```

Actualiza `app/generation/rag/store/models.py` para que el ORM conozca la columna y el índice. La
columna es de **solo lectura**: Postgres la mantiene a partir de `content`.

```python
from sqlalchemy import BigInteger, Computed, ForeignKey, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

# ...


class ChunkRow(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_chunk_type", "chunk_type"),
        Index("ix_chunks_metadata_gin", "metadata", postgresql_using="gin"),
        # Session 10: GIN over the generated tsvector that backs the lexical
        # branch of hybrid search.
        Index("ix_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
    )

    # ...

    # Session 10: STORED generated tsvector. ``english`` must match the
    # migration and ``websearch_to_tsquery`` in the repository, or Postgres bypasses
    # the GIN index and matching silently changes.
    content_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=True,
    )
```

Aplica la migración y verifica el esquema:

```bash
docker compose build api
docker compose run --rm api alembic upgrade head

docker compose exec estimator-postgres psql -U estimator -d estimator \
  -c "\d+ chunks" \
  -c "SELECT indexname FROM pg_indexes WHERE tablename = 'chunks';"
```

Debes ver `content_tsv` y el índice `ix_chunks_content_tsv`. La columna se rellena sola para las
filas existentes; no hace falta re-ingestar.

> **No añadas un trigger.** La columna `GENERATED ALWAYS ... STORED` es la forma declarativa de
> mantener el `tsvector` sincronizado. Un trigger sería una segunda fuente de verdad y una fuente
> de drift.

### Paso 2 — Rama léxica y fusión RRF

#### 2.1 — `search_lexical` en `ChunkStore`

En `app/generation/rag/store/repository.py`:

1. Extrae los filtros estructurales a un helper compartido por la búsqueda vectorial y la léxica.
2. Añade una constante `FTS_CONFIG = "english"` y un helper `_lexical_statement` (testeable sin
   Postgres).
3. Añade `search_lexical`, que devuelve `ChunkMatch` con la distancia centinela
   `LEXICAL_ONLY_DISTANCE = 1.0` (la rama léxica no tiene distancia coseno).

```python
from sqlalchemy import Integer, Select, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

# ...

# Session 10: text-search configuration. MUST match the generated column in
# migration 0003. Change both places together if the corpus language changes.
FTS_CONFIG = "english"

# A lexical-only hit never entered the vector ranking, so it has no cosine
# distance. 1.0 is a display sentinel ("far"); fusion and reranking only use
# the ordering, never this distance.
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


class ChunkStore:
    # ...

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
        """SELECT de la rama léxica: OR tsquery + ``ts_rank_cd``.

        ``websearch_to_tsquery`` recibe un string con OR entre términos para que
        un documento pueda coincidir con *algunos* keywords en vez de con todos.
        Para descripciones largas de proyecto, una consulta AND
        (``plainto_tsquery``) casi siempre devuelve cero filas y deja la rama
        híbrida inservible. ``ts_rank_cd`` (cover density) ordena por relevancia:
        mayor es mejor, lo contrario de la distancia coseno.
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
        """Rama léxica de la búsqueda híbrida (Sesión 10).

        Mismos filtros estructurales que ``search_with_filters`` para que las
        dos ramas vean el mismo espacio de candidatos. Devuelve como máximo
        ``top_k`` chunks ordenados por ``ts_rank_cd`` descendente.
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
        """Predicados compartidos por la rama vectorial y la léxica.

        Cada filtro sigue la convención ``None`` = no aplica. Sector y año se
        leen del JSONB; el tipo de chunk, de su columna.
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
```

Refactoriza `search_with_filters` para usar `self._structural_filters(...)` en lugar del bloque
inline. **No cambies su contrato**: sigue devolviendo `(matches, candidates_evaluated)`.

> **`websearch_to_tsquery` con OR, no `plainto_tsquery` con AND.** `plainto_tsquery` convierte el
> texto libre en un AND de todos sus lexemas: una consulta de proyecto de 20 palabras exige que el
> documento contenga las 20 y, en la práctica, devuelve cero filas. Para recall léxico queremos que
> un documento entre por coincidir con *algunos* términos; la precisión la resuelven RRF y el
> cross-encoder. Por eso `_or_tsquery_text` une los términos con `" or "` y se usa
> `websearch_to_tsquery`, que además no falla con sintaxis rara del usuario.

#### 2.2 — Fusión RRF

Crea `app/generation/rag/retrieval/fusion.py`:

```python
"""Reciprocal Rank Fusion (RRF) — combine independent rankings into one.

RRF fuses rankings by POSITION, not by score, which is exactly what we want when
the two branches produce incomparable scores: cosine distance (lower = better)
from the vector branch and ``ts_rank_cd`` (higher = better) from the lexical one.
Normalising those onto a shared scale is fiddly and corpus-dependent; RRF
sidesteps it by only trusting the order each branch puts things in.

The score of a document is ``sum(1 / (k + rank))`` over every ranking it appears
in (rank is 0-based here). The smoothing constant ``k`` damps the contribution of
top positions: a large ``k`` flattens the curve so a document must rank well in
*both* branches to win, while a small ``k`` lets a single #1 dominate. 60 is the
value from the original Cormack et al. paper and a sane default.

Pure and synchronous on purpose — trivially unit-testable, no I/O.
"""

from __future__ import annotations

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[int, float]]:
    """Fuse several id rankings into one, best first.

    Parameters
    ----------
    rankings:
        One list of chunk ids per branch, each already ordered best→worst. A
        branch that returned nothing contributes an empty list (it just adds no
        score). Duplicate ids within a single ranking are ignored after their
        first (best) occurrence.
    k:
        RRF smoothing constant (see module docstring). Must be positive.

    Returns
    -------
    list[tuple[int, float]]
        ``(chunk_id, fused_score)`` sorted by score descending. Ties break by
        ascending id so the output is deterministic.
    """
    if k <= 0:
        raise ValueError("RRF smoothing constant k must be positive")

    scores: dict[int, float] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for position, chunk_id in enumerate(ranking):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
```

> **Por qué RRF y no una media ponderada de scores.** La distancia coseno y `ts_rank_cd` no viven
> en la misma escala ni tienen la misma distribución. Normalizarlas exige decidir mínimos, máximos
> y pesos que dependen del corpus. RRF solo usa el orden: si un chunk es top-3 en ambas ramas,
> sube; si solo aparece en una, sube menos. Es la opción robusta y explicable.

### Paso 3 — Pipeline recall-then-rerank

Crea `app/generation/rag/retrieval/pipeline.py`. La función **no importa `app.dependencies`**: el
store, la factoría de sesiones y el reranker se inyectan desde el *composition root* o desde el
retriever. Así respetamos `tests/test_layering.py`.

```python
"""Recall-then-rerank retrieval pipeline (Session 10).

The single ``retrieve()`` entrypoint composes the four configurations the
exercise measures, behind two switches the caller resolves (request → runtime →
settings):

* ``search_mode="vector"`` — dense k-NN only (the Session 9 behaviour).
* ``search_mode="hybrid"`` — dense + lexical branches fused with RRF.
* ``rerank=False`` — keep the top ``top_k`` of the recall ordering.
* ``rerank=True``  — recall WIDE (``recall_k``, e.g. 50) then let the
  cross-encoder rescore and keep the top ``rerank_top_n`` (e.g. 5). The recall
  stage's only job is not to lose the relevant document; the reranker's job is
  to float it up.

It returns the same :class:`RetrievalOutcome` as ``retrieve_chunks`` so every
existing consumer (orchestrator, stage endpoints) is unaffected.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.generation.rag.retrieval.fusion import reciprocal_rank_fusion
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.schemas import RetrievalOutcome, RetrievedChunk
from app.generation.rag.store.repository import (
    LEXICAL_ONLY_DISTANCE,
    ChunkMatch,
    ChunkStore,
)

log = structlog.get_logger()


def _match_to_chunk(match: ChunkMatch, *, distance: float) -> RetrievedChunk:
    """Map a store DTO onto the typed retrieval contract."""
    raw_year = match.metadata.get("year")
    try:
        project_year = int(raw_year) if raw_year is not None else None
    except (TypeError, ValueError):
        project_year = None
    sector = match.metadata.get("client_sector")
    budget_id = match.metadata.get("budget_id")
    return RetrievedChunk(
        id=match.chunk_id,
        document_id=match.document_id,
        content=match.content,
        sector=str(sector) if sector is not None else None,
        project_year=project_year,
        chunk_type=match.chunk_type,
        distance=distance,
        budget_id=str(budget_id) if budget_id is not None else None,
    )


async def retrieve(
    *,
    store: ChunkStore,
    session_factory: async_sessionmaker[AsyncSession],
    query_embedding: list[float],
    query_text: str,
    search_mode: str = "vector",
    rerank: bool = False,
    top_k: int = 10,
    recall_k: int = 50,
    rerank_top_n: int = 5,
    distance_threshold: float = 0.6,
    rrf_k: int = 60,
    sectors: list[str] | None = None,
    project_year_min: int | None = None,
    project_year_max: int | None = None,
    chunk_types: list[str] | None = None,
    reranker: CrossEncoderReranker | None = None,
) -> RetrievalOutcome:
    """Run vector/hybrid retrieval with optional cross-encoder reranking.

    ``query_text`` is required for the lexical branch and the reranker even when
    ``search_mode`` is ``"vector"`` (the reranker scores against the raw query).
    ``reranker`` is injected by the caller; when ``rerank`` is True it must be
    present. Returns the top results best-first (cross-encoder order when
    reranking, else distance/RRF order).
    """
    if search_mode not in {"vector", "hybrid"}:
        raise ValueError(f"Unsupported search_mode: {search_mode!r}")
    if rerank and reranker is None:
        raise ValueError("rerank=True requires an injected reranker")

    started = time.perf_counter()

    # Recall wide whenever a later stage (fusion or rerank) will re-sort; recall
    # exactly top_k only for the plain vector path, where the order is final.
    wide = rerank or search_mode == "hybrid"
    vector_limit = recall_k if wide else top_k

    async with session_factory() as session:
        vector_matches, candidates_evaluated = await store.search_with_filters(
            session,
            query_vector=query_embedding,
            top_k=vector_limit,
            distance_threshold=distance_threshold,
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
        )
        lexical_matches: list[ChunkMatch] = []
        if search_mode == "hybrid":
            lexical_matches = await store.search_lexical(
                session,
                query_text=query_text,
                top_k=recall_k,
                sectors=sectors,
                project_year_min=project_year_min,
                project_year_max=project_year_max,
                chunk_types=chunk_types,
            )

    # Build the candidate pool once (id → chunk); vector distance wins over the
    # lexical sentinel when an id is in both branches.
    candidates: dict[int, RetrievedChunk] = {}
    for match in vector_matches:
        candidates[match.chunk_id] = _match_to_chunk(match, distance=match.distance)
    for match in lexical_matches:
        candidates.setdefault(
            match.chunk_id,
            _match_to_chunk(match, distance=LEXICAL_ONLY_DISTANCE),
        )

    # Recall ordering: RRF for hybrid, raw distance order for vector.
    if search_mode == "hybrid":
        fused = reciprocal_rank_fusion(
            [
                [match.chunk_id for match in vector_matches],
                [match.chunk_id for match in lexical_matches],
            ],
            k=rrf_k,
        )
        ordered = [candidates[chunk_id] for chunk_id, _score in fused]
    else:
        ordered = [candidates[match.chunk_id] for match in vector_matches]

    if rerank and reranker is not None:
        final = await asyncio.to_thread(
            reranker.rerank,
            query_text,
            ordered[:recall_k],
            top_n=rerank_top_n,
        )
    else:
        final = ordered[:top_k]

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "rag.retrieval.end",
        query=query_text[:80],
        search_mode=search_mode,
        rerank=rerank,
        vector_hits=len(vector_matches),
        lexical_hits=len(lexical_matches),
        results=len(final),
        candidates_evaluated=candidates_evaluated,
        store_search_mode=store.search_mode,
        search_time_ms=elapsed_ms,
    )
    return RetrievalOutcome(
        chunks=final,
        low_confidence=not final,
        candidates_evaluated=candidates_evaluated,
    )
```

> **`distance` después del rerank no es el score del cross-encoder.** El chunk conserva la
> distancia de recall (o `1.0` si solo lo encontró la rama léxica). El orden es el del
> cross-encoder. No interpretes `distance` como una probabilidad ni como el orden final.

> **El umbral de distancia solo se aplica a la rama vectorial.** Un hit léxico entra por coincidencia
> de términos aunque su distancia coseno sea alta. Es intencional: el recall léxico aporta
> candidatos que el vector no ve. Si quieres endurecerlo, hazlo en el reranker o añade un umbral
> léxico explícito y documentado.

#### 3.1 — Extensión de `SemanticRetriever`

En `app/generation/rag/retriever.py`, añade un método `retrieve` y haz que `retrieve_chunks`
delegue en él con `search_mode="vector"` y `rerank=False`. Así hay un único camino y el contrato de
la Sesión 09 no cambia.

```python
from app.generation.rag.retrieval.pipeline import retrieve as retrieve_pipeline
from app.generation.rag.retrieval.reranker import CrossEncoderReranker


class SemanticRetriever:
    def __init__(
        self,
        embedder: OpenAIEmbedder,
        session_factory: async_sessionmaker[AsyncSession],
        store: ChunkStore,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._session_factory = session_factory
        self._store = store
        self._reranker = reranker

    async def retrieve(
        self,
        *,
        query_text: str,
        search_mode: str = "vector",
        rerank: bool = False,
        top_k: int = 10,
        recall_k: int = 50,
        rerank_top_n: int = 5,
        distance_threshold: float = 0.6,
        rrf_k: int = 60,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
    ) -> RetrievalOutcome:
        query_embedding = await asyncio.to_thread(self._embedder.embed_one, query_text)
        return await retrieve_pipeline(
            store=self._store,
            session_factory=self._session_factory,
            query_embedding=query_embedding,
            query_text=query_text,
            search_mode=search_mode,
            rerank=rerank,
            top_k=top_k,
            recall_k=recall_k,
            rerank_top_n=rerank_top_n,
            distance_threshold=distance_threshold,
            rrf_k=rrf_k,
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
            reranker=self._reranker,
        )

    async def retrieve_chunks(
        self,
        *,
        query_text: str,
        top_k: int = 10,
        distance_threshold: float = 0.6,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
        search_mode: str = "vector",
        rerank: bool = False,
        recall_k: int = 50,
        rerank_top_n: int = 5,
        rrf_k: int = 60,
    ) -> RetrievalOutcome:
        """Retrieval with optional S10 toggles; defaults preserve S09.

        Mantener ``retrieve_chunks`` con parámetros opcionales evita romper a los
        consumidores y dobles de test de la Sesión 09, que ya llaman a este
        método. El camino nuevo ``retrieve`` sigue disponible para la API y la
        evaluación.
        """
        return await self.retrieve(
            query_text=query_text,
            search_mode=search_mode,
            rerank=rerank,
            top_k=top_k,
            recall_k=recall_k,
            rerank_top_n=rerank_top_n,
            distance_threshold=distance_threshold,
            rrf_k=rrf_k,
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
        )
```

Elimina `_to_retrieved_chunk` si queda duplicado: el aplanado vive ahora en
`pipeline._match_to_chunk`.

#### 3.2 — Integración en el pipeline grounded

En `app/generation/rag/grounded_estimator.py`, `_retrieve` debe leer los toggles efectivos y pasar
`search_mode`, `rerank`, `recall_k`, `rerank_top_n` y `rrf_k` al nuevo método. La estimación
end-to-end se beneficia de la configuración sin cambiar su contrato.

```python
async def _retrieve(transcript_brief: ProjectBrief) -> RetrievalOutcome:
    from app.dependencies import get_runtime_retrieval_config, get_semantic_retriever

    retriever = get_semantic_retriever()
    if retriever is None:
        raise RagRetrievalError("Embedding service is not configured.")

    retrieval_text = compose_retrieval_text(transcript_brief)
    sector = transcript_brief.sector.lower().strip() if transcript_brief.sector else None
    sectors = [sector] if sector in _KNOWN_SECTORS else None

    runtime = get_runtime_retrieval_config()
    search_mode = runtime.effective_search_mode()
    rerank = runtime.effective_rerank()

    try:
        return await retriever.retrieve_chunks(
            query_text=retrieval_text,
            search_mode=search_mode,
            rerank=rerank,
            top_k=settings.rag_retrieval_top_k,
            recall_k=settings.retrieval_recall_top_k,
            rerank_top_n=settings.rerank_top_n,
            distance_threshold=settings.rag_retrieval_distance_threshold,
            rrf_k=settings.rrf_k,
            sectors=sectors,
        )
    except Exception as exc:
        raise RagRetrievalError("Historical budget retrieval failed.") from exc
```

> El `except Exception` se mantiene por la misma razón que en la Sesión 09: la capa HTTP no debe
> filtrar detalles internos. El log estructurado ya registra `rag.retrieval.end` con el modo, el
> número de hits y la latencia.

### Paso 4 — Configuración y toggles sin tocar código

#### 4.1 — Settings

Los campos `retrieval_search_mode`, `reranker_enabled`, `reranker_model`,
`retrieval_recall_top_k`, `rerank_top_n` y `rrf_k` ya se añadieron en el **Paso 0.1**; el wrapper
los necesita para el gate. No los dupliques aquí. Limítate a comprobar que `Settings` los expone
con los defaults indicados y que `.env.example` está actualizado.

#### 4.2 — Override en Redis

En `app/foundation/llm/runtime_config.py`, añade un store separado del de modelos. Comparte el
patrón de `RuntimeModelConfig`: lectura degradada a settings si Redis no está, escritura que
propaga `RuntimeConfigUnavailable`.

```python
# --- Session 10: runtime overrides for retrieval toggles ---------------------
# Same Redis-hash override pattern as the model knobs, but for the two retrieval
# switches: search mode and reranking. A SEPARATE hash so resetting the model
# config does not clear them and vice versa.
RETRIEVAL_HASH_KEY = "estimator:runtime_retrieval"

SEARCH_MODE_KEY = "RETRIEVAL_SEARCH_MODE"
RERANK_KEY = "RERANKER_ENABLED"
RETRIEVAL_KEYS: tuple[str, ...] = (SEARCH_MODE_KEY, RERANK_KEY)

_VALID_SEARCH_MODES = ("vector", "hybrid")


class RuntimeRetrievalConfig:
    """Redis-hash-backed override store for the Session 10 retrieval toggles.

    Reads degrade to the settings default if Redis is down (the pipeline keeps
    serving exactly as .env configured it); writes re-raise so a failed override
    is visible to the API (mapped to 503), matching RuntimeModelConfig.
    """

    def __init__(self, redis_client: redis.Redis, settings: Settings) -> None:
        self._redis = redis_client
        self._settings = settings

    @classmethod
    def from_url(cls, url: str, settings: Settings) -> RuntimeRetrievalConfig:
        return cls(redis.from_url(url, decode_responses=True), settings)

    def _get_raw(self, key: str) -> str | None:
        try:
            return self._redis.hget(RETRIEVAL_HASH_KEY, key)
        except redis.RedisError as exc:
            log.warning("runtime_retrieval_read_failed", key=key, error=str(exc)[:200])
            return None

    def _set_raw(self, key: str, value: str | None) -> None:
        try:
            if value is None:
                self._redis.hdel(RETRIEVAL_HASH_KEY, key)
            else:
                self._redis.hset(RETRIEVAL_HASH_KEY, key, value)
        except redis.RedisError as exc:
            raise RuntimeConfigUnavailable(str(exc)) from exc

    def effective_search_mode(self) -> str:
        override = self._get_raw(SEARCH_MODE_KEY)
        if override in _VALID_SEARCH_MODES:
            return override
        return self._settings.retrieval_search_mode

    def effective_rerank(self) -> bool:
        override = self._get_raw(RERANK_KEY)
        if override is None:
            return self._settings.reranker_enabled
        return override.lower() == "true"

    def set_search_mode(self, value: str | None) -> None:
        if value is not None and value not in _VALID_SEARCH_MODES:
            raise ValueError(f"Invalid search mode: {value}")
        self._set_raw(SEARCH_MODE_KEY, value)

    def set_rerank(self, value: bool | None) -> None:
        self._set_raw(RERANK_KEY, None if value is None else str(value).lower())

    def snapshot(self) -> dict[str, dict[str, str | bool]]:
        """``{effective, default, overridden}`` per retrieval toggle."""
        try:
            overrides = self._redis.hgetall(RETRIEVAL_HASH_KEY)
        except redis.RedisError as exc:
            log.warning("runtime_retrieval_read_failed", key="*", error=str(exc)[:200])
            overrides = {}
        return {
            SEARCH_MODE_KEY: {
                "effective": self.effective_search_mode(),
                "default": self._settings.retrieval_search_mode,
                "overridden": SEARCH_MODE_KEY in overrides,
            },
            RERANK_KEY: {
                "effective": self.effective_rerank(),
                "default": self._settings.reranker_enabled,
                "overridden": RERANK_KEY in overrides,
            },
        }
```

Actualiza `__all__` con `RETRIEVAL_HASH_KEY`, `RETRIEVAL_KEYS`, `RuntimeRetrievalConfig`, etc.

#### 4.3 — Fábricas en `dependencies.py`

```python
from app.foundation.llm.runtime_config import RuntimeModelConfig, RuntimeRetrievalConfig
from app.generation.rag.retrieval.reranker import CrossEncoderReranker


@lru_cache
def get_runtime_retrieval_config() -> RuntimeRetrievalConfig:
    """Redis-backed override store for search mode + reranking.

    Read per call so a flip in the config API takes effect on the next
    retrieval without a restart.
    """
    settings = get_settings()
    return RuntimeRetrievalConfig.from_url(settings.redis_url, settings)


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    """Cross-encoder reranker singleton.

    Building the wrapper is cheap; the model loads lazily on the first rerank,
    so import time and app startup stay free of torch.
    """
    return CrossEncoderReranker.from_settings()
```

Actualiza `get_semantic_retriever` para inyectar el reranker:

```python
@lru_cache
def get_semantic_retriever() -> SemanticRetriever | None:
    embedder = get_embedder()
    if embedder is None:
        return None
    return SemanticRetriever(
        embedder=embedder,
        session_factory=get_async_session_factory(),
        store=get_chunk_store(),
        reranker=get_reranker(),
    )
```

Añade `get_runtime_retrieval_config.cache_clear()` y `get_reranker.cache_clear()` al fixture autouse
de `tests/conftest.py`, en el setup y en el teardown, como el resto de fábricas `lru_cache`.

#### 4.4 — Endpoint de retrieval

En `app/generation/rag/schemas.py`, añade los dos overrides opcionales a `RetrievalQuery`:

```python
class RetrievalQuery(BaseModel):
    """Payload de ``POST /api/v1/retrieval/search`` (S09 + S10 toggles)."""

    model_config = ConfigDict(extra="forbid")

    query_text: str = Field(min_length=10, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=30)
    distance_threshold: float = Field(default=0.6, ge=0.0, le=2.0)
    sectors: list[str] | None = None
    project_year_min: int | None = Field(default=None, ge=2010, le=2100)
    project_year_max: int | None = Field(default=None, ge=2010, le=2100)
    chunk_types: list[str] | None = None
    # Session 10 overrides. ``None`` falls back to runtime → settings.
    search_mode: Literal["vector", "hybrid"] | None = None
    rerank: bool | None = None
```

Añade `budget_id` a `RetrievedChunk` (opcional, retrocompatible). Es lo que permite medir precisión
por presupuesto padre:

```python
class RetrievedChunk(BaseModel):
    # ...
    budget_id: str | None = Field(
        default=None,
        description="Traceable corpus id of the parent budget (from JSONB metadata).",
    )
```

En `app/api/retrieval.py`, resuelve la precedencia **petición → runtime → settings** y llama al
nuevo método:

```python
from app.config import get_settings
from app.dependencies import get_runtime_retrieval_config, get_semantic_retriever


@router.post("/search", response_model=RetrievalOutcome)
@limiter.limit("120/minute")
async def search(
    request: Request,
    payload: RetrievalQuery,
    retriever: Annotated[SemanticRetriever | None, Depends(get_semantic_retriever)],
) -> RetrievalOutcome:
    """Busca chunks con modo vectorial/híbrido y reranking opcional."""
    if retriever is None:
        log.error("rag.retrieval.failed", reason="retriever_unavailable")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo ejecutar el retrieval.",
        )

    settings = get_settings()
    runtime = get_runtime_retrieval_config()
    search_mode = payload.search_mode or runtime.effective_search_mode()
    rerank = payload.rerank if payload.rerank is not None else runtime.effective_rerank()

    try:
        return await retriever.retrieve_chunks(
            query_text=payload.query_text,
            search_mode=search_mode,
            rerank=rerank,
            top_k=payload.top_k,
            recall_k=settings.retrieval_recall_top_k,
            rerank_top_n=settings.rerank_top_n,
            distance_threshold=payload.distance_threshold,
            rrf_k=settings.rrf_k,
            sectors=payload.sectors,
            project_year_min=payload.project_year_min,
            project_year_max=payload.project_year_max,
            chunk_types=payload.chunk_types,
        )
    except Exception as exc:
        log.error(
            "rag.retrieval.failed",
            reason="retrieval_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No se pudo ejecutar el retrieval.",
        ) from exc
```

#### 4.5 — Endpoints de configuración

En `app/api/config.py`, añade:

```python
from app.dependencies import get_runtime_config, get_runtime_retrieval_config
from app.foundation.llm.runtime_config import (
    RuntimeConfigUnavailable,
    RuntimeRetrievalConfig,
)


class RetrievalUpdateRequest(BaseModel):
    """Partial update of the Session 10 retrieval toggles.

    Only the keys present are touched; explicit ``null`` resets a knob to its
    .env default.
    """

    search_mode: str | None = Field(default=None, description="'vector' or 'hybrid'.")
    rerank: bool | None = Field(default=None, description="Enable cross-encoder reranking.")


@router.get("/retrieval")
def get_retrieval(
    runtime_retrieval: Annotated[RuntimeRetrievalConfig, Depends(get_runtime_retrieval_config)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Current retrieval configuration: effective/default/overridden per toggle."""
    return {
        "retrieval": runtime_retrieval.snapshot(),
        "reranker_model": settings.reranker_model,
    }


@router.put("/retrieval")
def update_retrieval(
    request: RetrievalUpdateRequest,
    runtime_retrieval: Annotated[RuntimeRetrievalConfig, Depends(get_runtime_retrieval_config)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Flip search mode and/or reranking at runtime — effective on the next call."""
    sent = request.model_fields_set
    try:
        if "search_mode" in sent:
            try:
                runtime_retrieval.set_search_mode(request.search_mode)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc
            log.info(
                "runtime_retrieval_changed",
                key="search_mode",
                new_value=request.search_mode,
            )
        if "rerank" in sent:
            runtime_retrieval.set_rerank(request.rerank)
            log.info("runtime_retrieval_changed", key="rerank", new_value=request.rerank)
    except RuntimeConfigUnavailable as exc:
        log.error("runtime_retrieval_write_failed", error=str(exc)[:200])
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El store de configuración runtime no está disponible.",
        ) from exc

    return {
        "retrieval": runtime_retrieval.snapshot(),
        "reranker_model": settings.reranker_model,
    }
```

Prueba los toggles sin reiniciar:

```bash
curl -s http://localhost:8000/api/v1/config/retrieval | jq
curl -s -X PUT http://localhost:8000/api/v1/config/retrieval \
  -H 'Content-Type: application/json' \
  -d '{"search_mode": "hybrid", "rerank": true}' | jq
curl -s -X PUT http://localhost:8000/api/v1/config/retrieval \
  -H 'Content-Type: application/json' \
  -d '{"search_mode": null, "rerank": null}' | jq
```

> **El endpoint de configuración no es UI.** Solo expone el store de overrides. No se toca
> `cliente-web` ni Streamlit; el ejercicio no requiere cambios de interfaz.

### Paso 5 — Golden set y medición

#### 5.1 — Golden set

Crea `evals/golden_retrieval.json` con el set de la sección **Datos para el ejercicio**. Revisa cada
consulta y cada `relevant_budget_ids`; si tu criterio cambia, anótalo y justifícalo en el documento
final. El campo `chunk_types: ["budget_component"]` es obligatorio para que la métrica no se mezcle
con el corpus task-level.

#### 5.2 — Script de evaluación

Crea `scripts/eval_retrieval_s10.py`:

```python
#!/usr/bin/env python3
"""Measure retrieval quality across the four Session 10 configurations.

Runs the golden set (``evals/golden_retrieval.json``) through the same
``retrieve()`` pipeline the API uses, for each of:

    A  Vector  / no rerank   (the Session 9 baseline)
    B  Hybrid  / no rerank
    C  Vector  / rerank
    D  Hybrid  / rerank

and reports precision@5 (mean over the queries) and query latency (mean over
measured runs) per configuration, plus a per-query precision breakdown.

Method notes:
* Queries are embedded ONCE upfront; embedding latency is an OpenAI round-trip
  shared by every config, so it is excluded from the timings (we measure the
  retrieval/rerank cost the techniques actually add).
* A permissive distance threshold is used so the top-5 is never truncated by the
  relevance floor — we are comparing RANKING quality, not the soft-fail gate.
* A retrieved chunk is relevant iff its parent ``budget_id`` is in the query's
  annotated ``relevant_budget_ids``. precision@5 = relevant_in_top5 / 5.

Usage (host, stack up + corpus ingested + OPEN_AI_KEY in .env)::

    uv run python scripts/eval_retrieval_s10.py

Reranking config runs download the cross-encoder weights on first use; verify
first with ``python -m app.generation.rag.retrieval.verify_reranker``.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rag_ops.common import require_embedder  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.dependencies import (  # noqa: E402
    get_async_session_factory,
    get_chunk_store,
    get_reranker,
)
from app.generation.rag.retrieval.pipeline import retrieve  # noqa: E402
from app.generation.rag.schemas import RetrievedChunk  # noqa: E402

GOLDEN_PATH = ROOT / "evals" / "golden_retrieval.json"

# (id, search label, rerank label, search_mode, rerank)
CONFIGS = [
    ("A", "Vector", "No", "vector", False),
    ("B", "Hybrid", "No", "hybrid", False),
    ("C", "Vector", "Yes", "vector", True),
    ("D", "Hybrid", "Yes", "hybrid", True),
]

# No effective relevance floor: we want a full top-k to grade ranking quality.
NO_FLOOR_THRESHOLD = 2.0
# Latency sampling per (query, config): one discarded warm-up + N measured runs.
MEASURED_RUNS = 3


def precision_at_k(chunks: list[RetrievedChunk], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k results whose parent budget is genuinely relevant."""
    top = chunks[:k]
    hits = sum(1 for chunk in top if chunk.budget_id in relevant_ids)
    return hits / k


async def _run_once(
    *,
    store: Any,
    session_factory: Any,
    query_embedding: list[float],
    query_text: str,
    search_mode: str,
    rerank: bool,
    settings: Any,
    chunk_types: list[str] | None,
    k: int,
    reranker: Any,
) -> Any:
    return await retrieve(
        store=store,
        session_factory=session_factory,
        query_embedding=query_embedding,
        query_text=query_text,
        search_mode=search_mode,
        rerank=rerank,
        top_k=k,
        recall_k=settings.retrieval_recall_top_k,
        rerank_top_n=k,
        distance_threshold=NO_FLOOR_THRESHOLD,
        rrf_k=settings.rrf_k,
        chunk_types=chunk_types,
        reranker=reranker,
    )


async def main() -> int:
    settings = get_settings()
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    queries = golden["queries"]
    chunk_types = golden.get("chunk_types")
    k = int(golden.get("k", 5))

    embedder = require_embedder()
    print(f"Embedding {len(queries)} golden queries (excluded from timings)...")
    embeddings = await asyncio.to_thread(
        lambda: {q["id"]: embedder.embed_one(q["query"]) for q in queries}
    )

    reranker = get_reranker()
    print("Warming up the cross-encoder (first load downloads weights)...")
    await asyncio.to_thread(reranker.load)

    store = get_chunk_store()
    session_factory = get_async_session_factory()

    results: dict[str, dict[str, Any]] = {
        cfg[0]: {"precisions": [], "latencies_ms": [], "per_query": {}} for cfg in CONFIGS
    }
    empty_warning = False

    for cfg_id, _search_label, _rerank_label, search_mode, rerank in CONFIGS:
        for query in queries:
            relevant = set(query["relevant_budget_ids"])
            embedding = embeddings[query["id"]]

            # Warm-up (discarded) then measured runs.
            await _run_once(
                store=store,
                session_factory=session_factory,
                query_embedding=embedding,
                query_text=query["query"],
                search_mode=search_mode,
                rerank=rerank,
                settings=settings,
                chunk_types=chunk_types,
                k=k,
                reranker=reranker,
            )
            samples: list[float] = []
            last = None
            for _ in range(MEASURED_RUNS):
                started = time.perf_counter()
                last = await _run_once(
                    store=store,
                    session_factory=session_factory,
                    query_embedding=embedding,
                    query_text=query["query"],
                    search_mode=search_mode,
                    rerank=rerank,
                    settings=settings,
                    chunk_types=chunk_types,
                    k=k,
                    reranker=reranker,
                )
                samples.append((time.perf_counter() - started) * 1000)

            if not last.chunks:
                empty_warning = True
            precision = precision_at_k(last.chunks, relevant, k)
            results[cfg_id]["precisions"].append(precision)
            results[cfg_id]["latencies_ms"].extend(samples)
            results[cfg_id]["per_query"][query["id"]] = precision

    _print_report(results, queries, k)
    if empty_warning:
        print(
            "\nWARNING: some configurations returned 0 chunks. Is the base corpus "
            "ingested? Run `docker compose run --rm api python scripts/query_examples.py`.",
            file=sys.stderr,
        )
    return 0


def _print_report(
    results: dict[str, dict[str, Any]],
    queries: list[dict[str, Any]],
    k: int,
) -> None:
    print(f"\n## Retrieval evaluation — precision@{k} and latency\n")
    print(f"| Config | Search | Reranking | Precision@{k} | Latency (ms) |")
    print("| --- | --- | --- | --- | --- |")
    for cfg_id, search_label, rerank_label, _mode, _rerank in CONFIGS:
        bucket = results[cfg_id]
        mean_precision = statistics.fmean(bucket["precisions"])
        mean_latency = statistics.fmean(bucket["latencies_ms"])
        print(
            f"| {cfg_id} | {search_label} | {rerank_label} | "
            f"{mean_precision:.2f} | {mean_latency:.1f} |"
        )

    print(f"\n### Per-query precision@{k}\n")
    header = "| Query | " + " | ".join(cfg[0] for cfg in CONFIGS) + " |"
    print(header)
    print("| --- | " + " | ".join("---" for _ in CONFIGS) + " |")
    for query in queries:
        row = [query["id"]] + [
            f"{results[cfg[0]]['per_query'][query['id']]:.2f}" for cfg in CONFIGS
        ]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

Ejecuta y guarda la salida real:

```bash
mkdir -p data/output/session10
docker compose exec api python scripts/eval_retrieval_s10.py \
  | tee data/output/session10/retrieval_eval.txt
```

También puedes ejecutarlo desde el host con `uv run python scripts/eval_retrieval_s10.py`.

> **Por qué se excluye el embedding de la latencia.** El round-trip a OpenAI es idéntico en las
> cuatro configuraciones y no depende de FTS, RRF ni del cross-encoder. Medirlo dentro mezclaría el
> coste del proveedor con el de las técnicas. Si quieres una columna end-to-end, añádela aparte y
> dilo en el documento; no cambies la comparación principal.

> **Por qué el umbral es 2.0.** Con un umbral de 0.6, una configuración puede devolver menos de 5
> chunks y la precisión@5 se confunde con el soft-fail. Para comparar *ranking* usamos un umbral
> permisivo y dejamos que el top-5 se llene. El soft-fail real se prueba aparte.

### Paso 6 — Documento de resultados y conclusiones

Crea `docs/sesion-10-busqueda-hibrida-reranking.md` con:

1. **Objetivo y alcance** — qué se mide y qué queda fuera.
2. **Diseño** — migración FTS, rama léxica, RRF (`k=60`), recall-then-rerank (`50 → 5`), toggles.
3. **Golden set y metodología** — las cinco consultas, criterio de relevancia por `budget_id`,
   embeddings fuera de la latencia, warm-up descartado, 3 ejecuciones medidas, umbral 2.0.
4. **Tabla comparativa A/B/C/D** — precisión@5 media y latencia media (y, si quieres, mediana).
5. **Desglose por consulta** — qué configuración gana en cada una y por qué.
6. **Evidencia** — comandos ejecutados y salida real (gate del reranker, migración, eval).
7. **Conclusiones** — un párrafo que responda:
   - ¿Qué configuración usarías en el proyecto y por qué?
   - ¿La ganancia de relevancia del reranking justifica su latencia en este caso de uso?
   - ¿Qué harías distinto a escala (caché de rerank, batch, modelo más pequeño, rerank solo en
     consultas de baja confianza…)?
8. **Limitaciones** — tamaño del corpus, una sola máquina, latencia CPU, golden set pequeño, etc.

No hay una respuesta única correcta. Lo que se evalúa es que la conclusión esté **argumentada con
los números que has medido**, no que elijas una configuración concreta.

## Entregable

Una rama nueva **`feature/pre-session-10`** con:

1. Migración `alembic/versions/0003_session10_fts.py`.
2. `app/generation/rag/retrieval/` completo (`__init__.py`, `fusion.py`, `reranker.py`,
   `pipeline.py`, `verify_reranker.py`).
3. `app/generation/rag/store/models.py` y `repository.py` actualizados.
4. `app/generation/rag/retriever.py` y `grounded_estimator.py` integrados.
5. `app/generation/rag/schemas.py` con `budget_id` y los toggles de `RetrievalQuery`.
6. `app/config.py`, `app/foundation/llm/runtime_config.py`, `app/dependencies.py`,
   `app/api/retrieval.py` y `app/api/config.py` actualizados.
7. `.env.example` con las variables nuevas.
8. `evals/golden_retrieval.json`.
9. `scripts/eval_retrieval_s10.py`.
10. `docs/sesion-10-busqueda-hibrida-reranking.md` con la tabla y las conclusiones.
11. `data/output/session10/retrieval_eval.txt` con la salida real.
12. Tests nuevos y actualizados.
13. `pyproject.toml` y `uv.lock` con `sentence-transformers`.
14. README actualizado con la sección de la Sesión 10 (comandos, variables y resultados).

Abre PR contra `main`. El PR puede quedar sin merge antes de la sesión en vivo; el branch basta.

## Criterios de "hecho"

El ejercicio está completo cuando:

- [ ] `docker compose exec api python -m app.generation.rag.retrieval.verify_reranker` termina en 0.
- [ ] `alembic upgrade head` aplica `0003_session10_fts` sin errores.
- [ ] La tabla `chunks` tiene `content_tsv` y el índice GIN `ix_chunks_content_tsv`.
- [ ] `ChunkStore.search_lexical` devuelve chunks ordenados por `ts_rank_cd` y respeta los filtros
      estructurales.
- [ ] `reciprocal_rank_fusion` está testeada con rankings vacíos, duplicados y empates.
- [ ] El pipeline implementa las cuatro configuraciones A/B/C/D y recuerda ancho (`recall_k=50`)
      cuando hay fusión o rerank.
- [ ] El reranker se inyecta desde `dependencies.py` y se carga de forma perezosa.
- [ ] `search_mode` y `rerank` se pueden fijar por petición, por runtime y por `.env`.
- [ ] El golden set tiene cinco consultas con `relevant_budget_ids` anotados.
- [ ] El script de evaluación produce precisión@5 y latencia para las cuatro configuraciones.
- [ ] El documento de resultados incluye la tabla, el desglose por consulta y las conclusiones.
- [ ] `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` y
      `uv run mypy app frontend` pasan.
- [ ] No hay cambios en `cliente-web`, Streamlit ni en los contratos existentes de la Sesión 09.

## Checklist de verificación

- [ ] La migración usa `to_tsvector('english', content)` y el modelo usa exactamente la misma
      expresión en `Computed(...)`.
- [ ] `websearch_to_tsquery` usa la misma configuración (`english`) que la columna generada.
- [ ] La rama léxica aplica los mismos filtros de sector/año/tipo que la vectorial.
- [ ] La fusión RRF usa `rank` 0-based y desempata por id ascendente.
- [ ] `rerank=True` recupera `recall_k` candidatos antes de puntuar, no `top_k`.
- [ ] El cross-encoder se ejecuta en un hilo (`asyncio.to_thread`) para no bloquear el event loop.
- [ ] El modelo no se importa a nivel de módulo: `sentence_transformers` se importa dentro de
      `_ensure_loaded`.
- [ ] Los tests del reranker usan un modelo falso y no descargan pesos.
- [ ] El golden set filtra `chunk_type="budget_component"` y no mezcla el corpus task-level.
- [ ] La latencia se mide con warm-up descartado y varias ejecuciones.
- [ ] La precisión@5 se calcula por `budget_id`, no por `chunk_id`.
- [ ] El documento final cita la latencia real medida y no números inventados.
- [ ] `git diff --stat` no muestra cambios en `cliente-web/` ni en `frontend/`.

## Troubleshooting

### `verify_reranker` falla al importar `sentence_transformers`

La dependencia no está en la imagen. Ejecuta `uv add sentence-transformers`, commitea `uv.lock` y
reconstruye:

```bash
docker compose build api
docker compose up -d
```

### `import torch` falla por `libgomp.so.1`

La imagen de runtime es `python:3.12-slim-bookworm` y algunas ruedas de PyTorch necesitan
`libgomp1`. Si el error aparece al cargar el modelo, añade la librería al stage `runtime` del
`Dockerfile`:

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```

### El modelo no descarga

- Comprueba la conectividad del contenedor y el proxy si lo usas.
- Revisa `RERANKER_MODEL` en `.env`; el default es
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
- La caché vive en `/home/app/.cache/huggingface` dentro del contenedor. Si el usuario `app` no
  puede escribir ahí, el fallo es de permisos.
- Si reconstruyes la imagen a menudo, monta un volumen para la caché de Hugging Face y evita
  descargar el modelo en cada build.

### La rama léxica no devuelve nada

- Comprueba que la migración está aplicada y que `content_tsv` no es `NULL`.
- Verifica que la configuración de la columna y la de `websearch_to_tsquery` coinciden.
- Prueba una consulta con palabras que existan literalmente en el corpus. Si el corpus es inglés y
  la query es española, el stemming no ayudará: es el límite del léxico, no un bug.
- Revisa los filtros: un sector o un `chunk_type` que no existe devuelve vacío.

### El reranker tarda demasiado

- La primera ejecución incluye la carga del modelo; no la cuentes como latencia.
- En CPU, un cross-encoder pequeño tarda decenas de milisegundos por lote de 50 pares. Si tu
  latencia es mucho mayor, revisa que no estés cargando el modelo en cada petición (singleton) ni
  ejecutándolo en el event loop.
- Para producción, considera rerank solo cuando la confianza del recall sea baja, caché por query o
  un modelo más pequeño. Son decisiones que puedes argumentar en las conclusiones.

### `content_tsv` no aparece en `\d+ chunks`

La imagen no tiene la migración nueva o Alembic no la aplicó. Reconstruye y ejecuta:

```bash
docker compose build api
docker compose run --rm api alembic upgrade head
```

## Documentación de referencia

- Rama de referencia del programa: `session_10` de
  [`LIDR-academy-temp/ai-engineering`](https://github.com/LIDR-academy-temp/ai-engineering/tree/session_10).
- PostgreSQL — Full Text Search (`tsvector`, `tsquery`, `ts_rank_cd`, GIN):
  https://www.postgresql.org/docs/current/textsearch.html
- PostgreSQL — Generated columns:
  https://www.postgresql.org/docs/current/ddl-generated-columns.html
- Cormack et al. — Reciprocal Rank Fusion:
  https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- sentence-transformers — CrossEncoder:
  https://www.sbert.net/docs/pretrained_cross-encoders.html
- Modelo `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`:
  https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- SQLAlchemy — `Computed` y columnas generadas:
  https://docs.sqlalchemy.org/en/20/core/defaults.html#sqlalchemy.schema.Computed
- Documentación interna de la Sesión 09:
  [`docs/sesion-9-fundamentos-rag-tecnicas-recuperacion.md`](../docs/sesion-9-fundamentos-rag-tecnicas-recuperacion.md)
- Persistencia pgvector de la Sesión 08:
  [`docs/persistencia-pgvector-busqueda-semantica-sesion-08.md`](../docs/persistencia-pgvector-busqueda-semantica-sesion-08.md)

## Nota

**Por qué híbrida y no solo vectorial.** Los embeddings capturan paráfrasis y sinónimos, pero se
diluyen con consultas largas o con vocabulario muy específico (“PSD2”, “HL7/FHIR”, “AGV”). La rama
léxica recupera coincidencias exactas que el vector puede dejar fuera. La combinación cubre dos
modos de fallo distintos: el vector falla en lo literal, el léxico falla en lo semántico.

**Por qué RRF y no una media de scores.** La distancia coseno y `ts_rank_cd` no son comparables.
Normalizarlas exige mínimos, máximos y pesos que dependen del corpus. RRF solo usa posiciones y es
robusto a cambios de escala. `k=60` es el valor clásico; con `k` pequeño un único primer puesto
domina, con `k` grande hace falta rankear bien en ambas ramas. Está como setting para que puedas
experimentar.

**Por qué recall-then-rerank.** Un cross-encoder es mucho más caro que un bi-encoder: puntúa pares
`(query, document)` y no puede precomputar los documentos. Por eso no se ejecuta sobre todo el
corpus, sino sobre un recall amplio (50) que ya trajo lo barato. La calidad del reranker depende de
que el documento relevante esté en ese recall; si no está, no hay reranker que lo salve. De ahí que
el recall sea ancho y el corte final, estrecho.

**Por qué el corpus en inglés manda.** El enunciado dice español, pero los datos reales de este
repo están en inglés. Usar `spanish` sobre un corpus inglés empeora el stemming y las stop-words.
La decisión se documenta en la migración y en el repositorio, y el cambio a `spanish` queda
explícito para el día en que el corpus cambie. Es preferible un sistema que funciona sobre sus
datos a uno que cumple literalmente un supuesto que los datos no cumplen.

**Qué es nuestro (y por qué).** Frente a la solución de referencia, este repo mantiene sus
contratos (`RetrievalOutcome`, `RetrievedChunk`, `ChunkMatch`, `RagRetrievalError`), su estructura
de capas (`app/generation/rag`, `app/api`, `app/foundation`), su composición en `dependencies.py` y
su servicio `api`. El pipeline no importa `app.dependencies`: recibe store, factoría de sesiones y
reranker por inyección, para que `tests/test_layering.py` siga siendo cierto. La migración usa
`english` porque el corpus es inglés. El golden set filtra `budget_component` porque el repo ya
tiene un corpus task-level. Y el documento de resultados es obligatorio: la sesión no va de “usar
híbrida porque mola”, va de **medir si compensa**.

**Qué NO hay que hacer.** No implementes el cross-encoder desde cero, no hagas fine-tuning, no
cambies la reformulación de queries de la Sesión 09, no añadas routing multi-índice ni filtros de
metadata nuevos. No toques la UI ni `cliente-web`. No inventes números de latencia ni de precisión:
la tabla tiene que salir del script contra tu corpus.
