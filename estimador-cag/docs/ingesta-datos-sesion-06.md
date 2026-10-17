# Ingesta de datos, persistencia y PII (Sesión 6)

> Cómo el estimator deja de ser puramente CAG y monta la puerta de entrada a RAG:
> auditar fuentes en un catálogo, convertirlas en `Document` canónicos, limpiar y
> validar datos tabulares, pseudonimizar PII y hacer tracking de los jobs en
> Postgres.

## Qué es y para qué sirve

Hasta la sesión 5 el estimator es un **CAG** (Cache-Augmented Generation): todo el
contexto cabe en el prompt *por construcción*. El stress test de la sesión 6 mide
dónde se rompe ese supuesto (latencia, coste y pérdida de memoria). La respuesta a
esos límites es **RAG**, y RAG empieza por la **ingesta**: convertir un corpus
heterogéneo (presupuestos JSON, transcripciones TXT, tarifas XLSX…) en documentos
canónicos, trazables y seguros de indexar.

Este documento describe el subsistema que hace eso. Sus piezas:

- **Catálogo** — auditoría versionada de las fuentes y de la decisión de ingerirlas
  o no (`include` / `review` / `exclude`).
- **Loaders + parsers** — bytes crudos → `list[Document]`.
- **Cleaning** — limpieza y validación de datos tabulares con pandas + pandera.
- **PII** — detección y pseudonimización reversible (GDPR) con Presidio + spaCy.
- **Persistencia** — Postgres (pgvector) + Alembic: tracking de jobs y mapping de
  PII.
- **API** — `POST /api/v1/ingestion/runs` (asíncrono) y `GET …/jobs/{id}`.

> **De CAG a RAG sin reescribir nada.** La ingesta es **aditiva**: no toca el
> camino de estimación (transaccional, conversacional ni ACB). Vive en
> `app/ingestion/` y `app/persistence/`, y su punto de entrada HTTP es un router
> nuevo. El CAG y el RAG conviven.

## Arquitectura y regla de dependencia

El subsistema sigue un estilo de **puertos y adaptadores** con una regla explícita:
el **núcleo** no conoce infraestructura.

| Capa | Responsabilidad | Ejemplos |
|---|---|---|
| Núcleo (caso de uso) | Orquestar la ingesta y validar el catálogo | `orchestrator.py`, `errors.py`, `ports.py` |
| Dominio | Modelos puros | `documents/models.py`, `catalog/models.py`, `parsers/protocol.py` |
| Adaptadores | Tocar el mundo exterior | `parsers/*`, `loaders/filesystem.py`, `cleaning/*`, `pii/*`, `persistence/*` |

El núcleo **no importa** SQLAlchemy, Presidio, pandas, pandera, Faker ni
`app.persistence`: depende de los puertos de `app/ingestion/ports.py`
(`JobTracker`, `IngestionUnitOfWork`), que los adaptadores satisfacen
estructuralmente. Esa frontera no es un comentario: la convierte en test
`tests/test_ingestion_layering.py` (falla si un fichero del núcleo importa una
librería prohibida).

## El pipeline

```
POST /api/v1/ingestion/runs {source_name}
  └→ router.ingestion            (valida, registra el job, responde 202, despacha)
       └→ orchestrator.ensure_ingestible(catalog, source_name)   ← validación ÚNICA
       └→ BackgroundTask → orchestrator.ingest_source(...)
             ├→ Catálogo            (¿existe? ¿decisión == include?)
             ├→ FileSystemLoader    (location → blobs del formato declarado)
             ├→ ParserRegistry      (formato → parser)
             └→ Document[]          (job: pending → running → completed | failed)
```

La decisión del catálogo se valida **una sola vez**, en `ensure_ingestible`:

- `UnknownSource` → HTTP **404** (`reason: unknown_source`).
- `SourceNotIncluded` → HTTP **400** (`reason: source_not_included`, `decision`).

El router sólo traduce esas excepciones de dominio a HTTP; no repite la lógica y
nunca importa parsers.

## El catálogo (`data/catalog/catalog.yaml`)

El catálogo **es código**: YAML validado contra modelos Pydantic
(`catalog/models.py`). Cada fuente declara owners, cadencia de refresco, scores de
calidad (1–5), sensibilidad (¿PII?), linaje y una **decisión explícita**:

- `include` — entra al pipeline.
- `review` — problemas conocidos, pendiente de sign-off del owner; se conserva,
  no se ingiere.
- `exclude` — fuera; exige un `decision_reason` (excluir es una decisión
  defendible, no un olvido).

```bash
# Validar el catálogo (imprime versión y fuentes incluidas)
uv run python -m app.ingestion.catalog.loader data/catalog/catalog.yaml

# Inspeccionar el corpus antes de decidir (facts del filesystem)
uv run python -m app.ingestion.catalog.inspect data/seed
```

## Loaders y parsers

- **`loaders/filesystem.py`** resuelve `CatalogSource.location` contra
  `INGESTION_DATA_ROOT`, recorre ficheros y emite `LoadedBlob` (bytes + ruta
  relativa). Una `location` que no existe es un error duro.
- **`parsers/protocol.py`** define el `Parser` (Protocol con
  `supported_formats` + `parse`) y el `ParseContext` (fuente, versión, timestamp).
- **`parsers/registry.py`** mapea formato → parser. En esta rama están
  registrados **JSON**, **TXT** y **CSV** (los parsers XLSX/DOCX/PDF son material
  de referencia y no se registran). El CSV (`csv_records.py`) es un parser propio
  añadido en la sesión: emite **un `Document` por fila** (columnas a
  `metadata.extra`), lo que demuestra que el contrato es extensible sin tocar el
  orquestador.

La salida es siempre un `Document` (`documents/models.py`): `id`, `text` y un
`DocumentMetadata` con `source_name`, `source_version`, `ingested_at`, linaje,
flags de sensibilidad y un `extra` libre por formato. El contrato es uniforme
entre formatos a propósito.

Añadir un parser es una línea: implementar el Protocol y
`registry.register(MiParser())`.

## Cleaning (datos tabulares)

`app/ingestion/cleaning/` limpia y valida la vista tabular de los presupuestos:

- **`budget_records.clean_budget_records()`** (pandas) normaliza nulls
  disfrazados (`"TBD"`, `"N/A"`, `""`), casing de moneda (`EUR`/`eur`), fechas
  mixtas, coerción numérica y **dedup por hash de contenido** (regla de negocio:
  conservar el `signed_at` más reciente).
- **`schemas.BudgetRecord`** (pandera) es el contrato de una fila limpia
  (`budget_id`, `currency`, `total_amount`, `signed_at`…).
- **`policy.validate_with_policy()`** reparte las filas que fallan en tres cubos:
  `valid`, `quarantined` (recuperables) y `discarded` (estructuralmente mal).

`cleaning` **no está cableado** en el endpoint de ingesta: se ejercita desde
`scripts/demo_cleaning_s06.py` y sus tests. Es una etapa de análisis sobre el
contenido parseado, antes de cualquier embedding.

## PII (pseudonimización reversible)

`app/ingestion/pii/` detecta y enmascara datos personales:

- **`analyzer.build_analyzer()`** inicializa Presidio contra el modelo spaCy
  **`es_core_news_md`**. Sin un modelo español explícito, Presidio usa inglés y
  **omite** los nombres hispanos sin avisar.
- **`recognizers.py`** añade reconocedores de dominio por regex: `BUDGET_ID`
  (`BUDGET-YYYY-NNNN`) y `CLIENT_CODE` (`CLI-NNNN`).
- **`pseudonymizer.ConsistentPseudonymizer`** reemplaza cada entidad por un
  pseudónimo estable: HMAC-SHA256 del valor original (con salt de servidor) +
  Faker `es_ES`. *Mismo valor original → mismo pseudónimo*, siempre.
- **`mapping_store`** persiste el mapping (`PostgresMappingStore`) o lo guarda en
  memoria (`InMemoryMappingStore`, para tests/scripts). En la BD sólo entra el
  **hash**, nunca el valor en claro: borrar una fila por hash revoca la
  reversibilidad (GDPR art. 17) sin tocar el corpus.

```bash
uv run python scripts/demo_pii_s06.py
```

**Cableado en el pipeline (Sesión 6).** La pseudonimización no es sólo una
utilidad offline: si `INGESTION_PSEUDONYMIZE_PII=true` y la fuente declara
`sensitivity.has_pii`, `ingest_source` sustituye el `text` de cada `Document` por
su versión pseudonimizada antes de completar el job. El núcleo no conoce Presidio:
depende del puerto `TextPseudonymizer` (`__call__: str → str`), y la frontera
construye el adaptador sobre la sesión del UoW (`get_text_pseudonymizer`). El
modelo spaCy sólo se carga cuando alguna fuente con PII se ingiere.

> **Límite conocido.** La consistencia es por `(entity_type, valor)`: el mapping
> se indexa por hash. El modelo `es_core_news_md` puede clasificar un mismo texto
> como `PERSON` o `LOCATION` según el contexto (p. ej. "Laura Fernández" tras
> "De nuevo"), y entonces el pseudónimo difiere. Es una limitación del modelo, no
> del pseudonimizador.

## Persistencia y migraciones

Persistencia con **SQLAlchemy 2.0 síncrono** (la ingesta no está en el camino
caliente del usuario) y **Alembic**:

- Tabla **`ingestion_jobs`** — tracking de los jobs (estado, `documents_count`,
  error, timestamps).
- Tabla **`pseudonym_mappings`** — mapping de PII (`entity_type`, `original_hash`,
  `pseudonym`), con unique constraint sobre `(entity_type, original_hash)`.

**Unit of Work ligero.** Los repositorios sólo hacen `flush()`; el `commit()` lo
decide la **frontera** (`UnitOfWork`): el router tras crear el job y el caso de
uso en cada transición de estado. Así la atomicidad queda bajo control y los tests
pueden sustituir el UoW por un doble en memoria.

```bash
# Aplicar el esquema
uv run alembic upgrade head
# Generar el SQL sin conexión (útil en CI)
uv run alembic upgrade head --sql
```

> **Dos Postgres, a propósito.** `estimator-postgres` (imagen **pgvector**,
> puerto host **5433**) sirve al subsistema de ingesta; `postgres` (alpine, 5432)
> sirve a `cliente-web`. No se mezclan. La extensión pgvector aún no se activa
> (llega en la sesión 7); en esta rama sólo se usan columnas normales.

## API de ingesta

```bash
# Catálogo de fuentes (con filtro opcional por decisión)
http GET :8000/api/v1/ingestion/sources decision==include

# Lanza una ingesta (202 + job_id). La fuente debe estar `include` en el catálogo.
http POST :8000/api/v1/ingestion/runs source_name=presupuestos_json

# Listado paginado de jobs, del más reciente al más antiguo
http GET :8000/api/v1/ingestion/jobs limit==5 offset==0

# Estado de un job
http GET :8000/api/v1/ingestion/jobs/<job_id>
```

La respuesta 202 devuelve `status=pending`; el `BackgroundTask` avanza el job a
`running` y, al terminar, `completed` (con `documents_count`) o `failed` (con
`error_message`). `sources` responde 422 si el filtro `decision` no es uno de
`include|review|exclude`.

## CLI de decisión arquitectónica

`app/ingestion/architecture.py` es la forma ejecutable de la decisión CAG / RAG /
Híbrido: evalúa cuatro restricciones del CAG (ventana, coste, latencia,
lost-in-the-middle) y cuatro ejes de decisión (volumen, frecuencia,
trazabilidad, control de acceso).

```bash
uv run python -m app.ingestion.architecture
```

## Ficheros y configuración

```
app/ingestion/
  architecture.py            CLI CAG/RAG/Hybrid
  orchestrator.py            caso de uso ingest_source + ensure_ingestible
  errors.py                  UnknownSource, SourceNotIncluded
  ports.py                   JobTracker, IngestionUnitOfWork, TextPseudonymizer (Protocols)
  catalog/                   models, loader, inspect
  loaders/filesystem.py      FileSystemLoader + LoadedBlob
  parsers/                   protocol, registry, budget_json, transcript_txt, csv_records
  documents/models.py        Document, DocumentMetadata
  cleaning/                  budget_records, schemas (pandera), policy
  pii/                       analyzer, recognizers, pseudonymizer, mapping_store
app/persistence/
  database.py                engine + SessionLocal + get_session
  models.py                  IngestionJobRow, PseudonymMappingRow
  unit_of_work.py            UnitOfWork (UoW ligero)
  repositories/              jobs, mappings
app/routers/ingestion.py     endpoints HTTP
app/schemas/ingestion.py     contratos Pydantic
alembic/                     env.py + versions/0001_session6_initial.py
data/catalog/catalog.yaml    catálogo de fuentes
data/seed/                   corpus (budgets/, transcripts/, presupuestos_extra.csv, rate_card_2024.xlsx)
scripts/                     demo_cleaning_s06, demo_pii_s06, preflight_s06
```

La pantalla **/ingestion** del cliente de negocio (`cliente-web/`) consume estos
endpoints (catálogo, lanzar ingesta y estado de jobs) sin persistir nada: la API
Python es la dueña de los jobs.

Variables de entorno (Sesión 6):

| Variable | Default | Notas |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://estimator:estimator@localhost:5433/estimator` | En compose apunta a `estimator-postgres:5432`. |
| `CATALOG_PATH` | `data/catalog/catalog.yaml` | Relativo a la raíz del proyecto si no es absoluto. |
| `INGESTION_DATA_ROOT` | `data/seed` | Raíz de resolución de `location`. |
| `PRESIDIO_SPACY_MODEL` | `es_core_news_md` | Modelo de Presidio (instalado como wheel fija). |
| `PSEUDONYM_FAKER_LOCALE` | `es_ES` | Locale de Faker. |
| `PSEUDONYM_HASH_SALT` | `change-me-in-prod` | Sal del HMAC; **cambiar en producción**. |
| `INGESTION_PSEUDONYMIZE_PII` | `true` | Pseudonimiza el texto de las fuentes con PII al ingerir. |

## Cómo se prueba

```bash
uv run pytest -v
uv run ruff check . && uv run ruff format --check .
uv run mypy app frontend
```

Tests específicos de la sesión 6:

- `test_ingestion_endpoint.py` — router + orquestador con un UoW en memoria
  (sin Postgres): 202/pending, completed, 404 (desconocida), 400 (excluida),
  `GET /sources` (con filtro) y `GET /jobs` (paginado).
- `test_ingestion_layering.py` — la regla de dependencia del núcleo.
- `test_ingestion_pii_pipeline.py` — el cableado del pseudonimizador en
  `ingest_source` (con un pseudonimizador falso).
- `test_catalog_loader.py`, `test_parsers_json.py`, `test_parsers_txt.py`,
  `test_parsers_csv.py`, `test_architecture.py` — catálogo y parsing.
- `test_cleaning_budgets.py`, `test_validation_policy.py` — limpieza y política.
- `test_pii_recognizers.py`, `test_pii_pseudonymizer.py` — detección (stub) y
  consistencia del pseudonimizador.

## Cómo ejecutarlo con Docker

```bash
# Levanta api + redis + estimator-postgres (+ ui + cliente-web)
docker compose up --build

# El servicio `api` aplica las migraciones al arrancar (alembic upgrade head).
curl -X POST http://localhost:8000/api/v1/ingestion/runs \
  -H 'Content-Type: application/json' \
  -d '{"source_name": "presupuestos_json"}'

# Pre-flight de la sesión (paquetes, modelo spaCy, corpus, catálogo, /health, BD)
docker compose exec api python scripts/preflight_s06.py
```

## Decisiones de diseño y límites

- **UoW ligero** (repos `flush()`, commit en la frontera): atomicidad explícita y
  tests sin base de datos.
- **Validación única**: la decisión del catálogo vive en `ensure_ingestible`; el
  router sólo mapea errores.
- **Regla de dependencia ejecutable**: el núcleo no importa infraestructura
  (`test_ingestion_layering.py`).
- **Síncrono a propósito**: la ingesta corre en `BackgroundTask`; no se introduce
  SQLAlchemy async para mantener simple la infraestructura docente.
- **PII cableada; cleaning fuera del endpoint**: `ingest_source` pseudonimiza las
  fuentes con PII (puerto `TextPseudonymizer`), pero `cleaning` sigue siendo una
  etapa de análisis que se ejercita por scripts y tests.
- **Parsers XLSX/DOCX/PDF no registrados** en esta rama (material de referencia);
  `unstructured`/`openpyxl` no son dependencias de runtime. El parser **CSV** sí
  está registrado y es propio.
