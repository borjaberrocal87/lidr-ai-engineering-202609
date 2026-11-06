# Sesión 9 — RAG end-to-end sobre transcripciones

Esta sesión cierra el bucle que faltaba desde el primer día: una **transcripción
de reunión** entra, y sale una **estimación en engineer-days respaldada por
presupuestos históricos**, con módulos, tareas y citas.

La diferencia con la Sesión 8 es de contrato: allí el retrieval devolvía chunks
similares; aquí el retrieval **gobierna** la generación. Cada cifra debe poder
trazarse a un `<source id="…">` recuperado. Si el contexto no alcanza, el
sistema debe decirlo (`confidence="insufficient"`) en vez de inventar.

## 1. Pipeline

```text
transcripción
  → brief estructurado (ProjectBrief)
  → texto de búsqueda
  → embedding
  → retrieval filtrado + umbral (soft-fail)
  → truncado por tokens (chunks completos)
  → contexto XML <source>
  → generación grounded (GroundedEstimate)
  → validación de citas (1 reintento correctivo)
  → coherencia de insuficiencia (1 reparación)
  → idempotencia
```

| Etapa | Módulo |
|---|---|
| Comprensión | `app/generation/rag/brief_extractor.py` |
| Retrieval | `app/generation/rag/retriever.py` + `store/repository.py` |
| Aumentación | `app/generation/rag/context_builder.py` |
| Generación | `app/generation/rag/grounded_estimator.py` |
| Grounding | `app/generation/rag/grounding.py` |
| Prompts | `app/foundation/prompts/rag_estimation/v1/` |
| Observabilidad | `app/generation/rag/stage_logging.py` |
| Idempotencia | `app/generation/rag/idempotency.py` |

### Por qué no se embebe la transcripción

Una transcripción tiene digresiones, anécdotas y ruido conversacional. Embeberla
produce un vector que es la media semántica de varias intenciones: distancias
comprimidas y hits irrelevantes. Por eso el primer paso extrae un
`ProjectBrief` (qué construir, con qué tecnología, sector, país, regulaciones y
restricciones) y compone con él una frase técnica corta.

## 2. Contrato de salida

`GroundedEstimate` es **independiente** del `EstimationResult` de la Sesión 4:

- `total_engineer_days`, `duration_weeks`.
- `modules[]` → `tasks[]` con `engineer_days` y `sources` (ids de chunk).
- `sources[]` con `relevance` y `used_for`.
- `assumptions[]` para alcance sin análogo histórico.
- `confidence`: `high | medium | low | insufficient`.
- `reasoning` y `insufficient_context_explanation`.

Regla dura: si `confidence="insufficient"`, los totales son `null`, `modules`
está vacío y la explicación es obligatoria. La guarda vive en
`grounding.is_coherent_estimate`.

## 3. Endpoints

Todos cuelgan de `/api/v1`.

| Método | Ruta | Auth | Límite |
|---|---|---|---|
| POST | `/retrieval/search` | `X-API-Key` (retrieval) | 120/min |
| POST | `/estimate/from-transcript` | `X-API-Key` (estimate) | 10/min |
| POST | `/estimate/stages/brief` | `X-API-Key` (estimate) | 30/min |
| POST | `/estimate/stages/retrieve` | `X-API-Key` (estimate) | 60/min |
| POST | `/estimate/stages/assemble` | `X-API-Key` (estimate) | 60/min |
| POST | `/estimate/stages/generate` | `X-API-Key` (estimate) | 15/min |

- Si la API key esperada no está configurada, el endpoint responde **401**.
- Los errores del pipeline se traducen a **502**; el detalle queda en los logs
  y el `request_id` viaja en `X-Request-ID`.
- El rate limit devuelve **429** con `Retry-After`.

### Retrieval

`POST /api/v1/retrieval/search` acepta `query_text`, `top_k`,
`distance_threshold`, `sectors`, `project_year_min`, `project_year_max` y
`chunk_types`. Los filtros se aplican en SQL antes del ranking vectorial:

- `client_sector` y `year` viven en JSONB.
- `chunk_type` es columna tipada.
- La distancia es coseno y respeta el modo `halfvec` de la Sesión 8.

Si ningún chunk cruza el umbral, devuelve `low_confidence=true` y una lista
vacía: es un **soft-fail**, no un error.

## 4. Configuración

En `.env` (todos configurables):

```text
RAG_BRIEF_MODEL=gpt-5-mini
RAG_BRIEF_MAX_TOKENS=8192
RAG_GENERATION_MODEL=gpt-5
RAG_GENERATION_REASONING_EFFORT=high
RAG_GENERATION_MAX_TOKENS=64000
RAG_RETRIEVAL_TOP_K=10
RAG_RETRIEVAL_DISTANCE_THRESHOLD=0.6
RAG_MAX_CONTEXT_TOKENS=30000
RAG_IDEMPOTENCY_TTL=86400
LLM_TIMEOUT_SECONDS=180
RETRIEVAL_API_KEY=dev-retrieval-key
ESTIMATE_API_KEY=dev-estimate-key
```

Los modelos de razonamiento (familia `gpt-5`) consumen tokens de razonamiento
del presupuesto de salida; por eso `RAG_GENERATION_MAX_TOKENS` es alto. La
generación puede tardar más de 30 s, de ahí `LLM_TIMEOUT_SECONDS=180`.

El wrapper filtra los parámetros opcionales por modelo: no envía `temperature`
a `gpt-5`/o1/o3/o4 salvo que sea `1.0`, y no envía `reasoning_effort` a modelos
que no lo soportan (`gpt-4o`, `gpt-4o-mini`). Un timeout no degrada el formato
de salida; se relanza para no multiplicar la espera.

Antes de enviar el JSON schema a OpenAI, `_response_format` lo normaliza al
subconjunto estricto: `additionalProperties: false` en cada objeto, todas las
propiedades en `required` y eliminación de keywords que el proveedor rechaza
(`default`, `minimum`, `maxLength`…). La validación real sigue siendo Pydantic
después de la respuesta.

## 5. Corpus task-level

El corpus de la Sesión 8 tiene un chunk por *componente*. La estimación S09 es
módulo → tarea, así que añadimos un corpus con esa granularidad:

```bash
uv run python scripts/build_task_corpus.py --generate-only
uv run python scripts/build_task_corpus.py --ingest
```

- `data/task_corpus.json`: 12 proyectos / 222 tareas, determinista.
- `document_type='historical_task_breakdown'`.
- `chunk_type='historical_task'`.
- Cada tarea lleva `module` en el texto y en la metadata del chunk.

La ingesta es idempotente: un 409 significa que el documento ya estaba.

## 6. Wizard en `cliente-web`

Ruta base: `/rag-estimations` (Express + EJS + Tailwind).

| Método | Ruta |
|---|---|
| GET | `/rag-estimations` |
| GET | `/rag-estimations/new` |
| POST | `/rag-estimations` |
| GET | `/rag-estimations/:id?step=brief\|retrieval\|assembly\|generation\|verification` |
| POST | `/rag-estimations/:id/brief` |
| POST | `/rag-estimations/:id/retrieve` |
| POST | `/rag-estimations/:id/assemble` |
| POST | `/rag-estimations/:id/generate` |
| POST | `/rag-estimations/:id/verify` |

Cada run persiste la transcripción y el resultado de cada etapa en
`rag_estimations` (migración `003_create_rag_estimations.sql`), de modo que se
puede revisar, re-ejecutar una etapa y verificar sin volver a pagar las
anteriores. La verificación permite editar módulos, tareas, días y notas; el
total se recalcula al guardar.

El cliente envía `X-API-Key` con `RAG_RETRIEVAL_API_KEY` o
`RAG_ESTIMATE_API_KEY` según la superficie.

## 7. Tests

```bash
cd estimador-cag
uv run pytest
uv run ruff check .
uv run mypy app

cd cliente-web
npm test
npm run build:css
```

Cobertura S09:

- `tests/test_rag_schemas.py`
- `tests/test_brief_extractor.py`
- `tests/test_context_builder.py`
- `tests/test_grounding.py`
- `tests/test_rag_estimator.py`
- `tests/test_retrieval_filtered.py`
- `tests/test_rag_security.py`
- `tests/test_rag_stages.py`
- `tests/test_task_corpus.py`
- `cliente-web/test/ragEstimationClient.test.js`
- `cliente-web/test/ragEstimations.test.js`

## 8. Límites y siguientes pasos

- El corpus task-level es sintético y determinista; no sustituye a presupuestos
  reales etiquetados.
- El retrieval no reordena con un cross-encoder; el umbral de coseno filtra,
  pero no rerankea.
- La verificación humana no reentrena ni recalibra el modelo: solo ajusta el
  resultado del run.
- Los prompts viven en `v1`; cambiar la política de grounding es añadir
  `v2` y pasar la versión, no editar el módulo.
