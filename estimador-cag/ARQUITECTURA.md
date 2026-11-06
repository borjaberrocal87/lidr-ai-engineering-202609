# Arquitectura del estimador

Este documento es el **contrato de capas** del servicio. Cualquier pieza nueva debe encajar en
una de estas capas; si no encaja, primero se decide dónde vive (y se actualiza este documento),
nunca se crea otra carpeta suelta en la raíz. Las reglas se verifican en
`tests/test_layering.py`.

## 1. Las tres arquitecturas de IA

El estimador no es un único patrón de generación: apila tres, de forma aditiva, sobre una base
común.

- **CAG — Cache-Augmented Generation** (`app/generation/cag/`). Responde sin tocar el LLM cuando
  ya hay una respuesta equivalente: primero un acierto exacto (SHA-256), luego uno por similitud
  vectorial.
- **RAG — Retrieval-Augmented Generation** (`app/generation/rag/`). Convierte presupuestos
  históricos en chunks + embeddings, los persiste en pgvector y los recupera para enriquecer el
  prompt con conocimiento citable.
- **Agentic** (`app/generation/agentic/`). Un bucle Actor-Critic-Boss que itera y audita la
  estimación antes de aceptarla, apoyado en la conversación multi-turno
  (`app/generation/conversation/`).

**Tesis central:** estas tres capas no se conocen entre sí. Componen **únicamente** a través del
conductor (`app/domain/estimation_service.py`). Esa es la regla que evita que el proyecto vuelva
a degenerar en una sucesión de carpetas acopladas.

## 2. El pastel de capas

```text
app/
├── main.py                         # factory FastAPI + lifespan
├── config.py                       # settings singleton (get_settings, @lru_cache)
├── dependencies.py                 # composition root: wiring de singletons
│
├── foundation/                     # plomería sin opinión de arquitectura AI
│   ├── llm/                        # LLMWrapper + runtime_config + errores de proveedor
│   ├── prompts/                    # loader Jinja2 + plantillas versionadas
│   ├── guardrails/                 # input (moderación + injection + PII) / output (scope)
│   ├── attachments/                # extracción de texto de PDF/DOCX
│   ├── cache/                      # backends genéricos (memoria, Redis, null)
│   ├── persistence/                # engine + modelos + repositories + unit_of_work
│   └── observability/              # logging structlog
│
├── domain/                         # contrato + conductor
│   ├── schemas/                    # Pydantic: estimations, acb, critic, conversation, ingestion…
│   ├── ports/                      # puertos compartidos (JobTracker, UoW, pseudonymizer)
│   ├── estimation_service.py       # EstimationService — ÚNICO punto de composición
│   └── streaming_service.py        # camino texto libre / SSE
│
├── generation/                     # las 3 arquitecturas + substrato conversacional
│   ├── cag/                        # exact.py + semantic.py
│   ├── rag/                        # chunking/ + embedding/ + analysis/ + store/ + ingest_service.py
│   ├── agentic/                    # boss.py + critic.py
│   └── conversation/               # models, store, metadata_extractor, tier_resolver, compression/
│
├── ingestion/                      # pipeline batch (offline) de fuentes
│   ├── catalog/ loaders/ parsers/ cleaning/ pii/ documents/
│   └── orchestrator.py · architecture.py
│
└── api/                            # transporte: routers finos, sin lógica de negocio
    ├── estimations.py · sessions.py · ingestion.py
    └── search.py · embeddings.py · config.py
```

## 3. Reglas de dependencias

| Capa | PUEDE importar | NO PUEDE importar |
|---|---|---|
| `config.py` | (nada interno) | — |
| `domain/schemas`, `domain/ports` | stdlib, Pydantic, `config` | `foundation`, `generation`, `api`, servicios del `domain` |
| `foundation/*` | `config`, `domain/schemas`, `domain/ports` | `generation`, `api`, `ingestion`, servicios del `domain` |
| `generation/<x>/*` | `config`, `foundation`, `domain/schemas`, `domain/ports` | `api`, `dependencies`, **otro hermano de `generation`** |
| `domain/estimation_service.py` (CONDUCTOR) | todos los hermanos de `generation` + `foundation` + contratos | `api` |
| `ingestion/*` | `config`, `foundation`, `domain/ports`, `generation/rag` | `api`, conductor |
| `api/*` | `dependencies`, `domain`, `foundation` (errores), tipos de `generation` | lógica de negocio |
| `dependencies.py` (COMPOSITION ROOT) | cualquier cosa | (lo importan solo `api/`, `main`, los tests y los conductores) |
| `main.py` | `api`, `config`, `foundation/observability` | — |

Tres aristas especiales, explícitas:

1. `generation/agentic` **puede** importar `generation/conversation` (lo agéntico se construye
   sobre el multi-turno). La inversa está **prohibida**.
2. `foundation/*` **puede** importar los contratos puros (`domain/schemas`, `domain/ports`) para
   que prompts y guardrails no tengan que duplicar el vocabulario del dominio.
3. Los dos conductores (`domain/estimation_service.py` y `domain/streaming_service.py`) usan
   `app.dependencies` de forma perezosa dentro de los métodos. Es una excepción controlada:
   el composition root nunca importa el conductor en tiempo de import.

## 4. El conductor

`app/domain/estimation_service.py::EstimationService` es el único sitio donde se cablean las
capas. Sus tres entradas y qué tocan:

- `estimate()` — guardrails(in) → **cag** (exacto + semántico) → prompts → **llm** → guardrails(out) → cag.store.
- `estimate_conversational()` — **conversation** (historial + metadata + compresión) → prompts → llm → guardrails(out).
- `estimate_with_acb()` — **agentic** (Boss orquesta Actor + Critic sobre la conversación).

Cuando RAG entre en el camino de petición (S09+), el paso de retrieval se añade **aquí**, como
un step más del conductor, no dentro de otra capa. El conductor es también donde se unen CAG y
RAG si una misma respuesta debe consultar caché antes de recuperar contexto.

## 5. Composition root

`dependencies.py` y `config.py` viven en la raíz, por encima de las capas, a propósito: el
composition root tiene permiso para alcanzar cualquier capa (es su trabajo cablear). Todas las
fábricas de singletons (`get_llm_wrapper`, `get_cache`, `get_semantic_cache`,
`get_session_store`, los chunkers, `get_estimation_service`) están en `dependencies.py`.

## 6. Camino de la petición principal

```text
POST /api/v1/estimate
└→ app/api/estimations.py (HTTP fino, mapeo de errores)
└→ app/domain/estimation_service.py::estimate()
   1. app/foundation/guardrails/input.py (moderación + injection + PII)
   2. app/generation/cag/exact.py (acierto exacto SHA-256)
   3. app/generation/cag/semantic.py (similitud vectorial redisvl)
   4. app/foundation/prompts/loader.py (Jinja2 versionado)
   5. app/foundation/llm/wrapper.py (Instructor + validators + re-prompt)
   6. app/foundation/guardrails/output.py (filter de scope)
   7. cag.exact.set() + cag.semantic.store()
   8. return EstimationResponse(result, prompt_version, cached)
```

## 7. Dos pipelines de ingesta, una decisión

- `app/ingestion/` es la **ingesta batch offline de fuentes**: catálogo, loaders, parsers,
  cleaning, PII, documentos y jobs. No es el camino de escritura del corpus RAG.
- `app/generation/rag/` es el **camino RAG**: chunking, embeddings, store pgvector,
  `RagIngestService` y `SemanticRetriever`.

Se mantienen separados a propósito. La referencia de la sesión también los separa: `ingestion/`
es una capa propia y `generation/rag/` no la absorbe. El diagnóstico de S09 ya señalaba que
compartir un único pipeline es trabajo futuro, no un requisito del refactor.

## 8. ¿Dónde va mi código nuevo?

| Si añades… | Va en… |
|---|---|
| Backend LLM, plantilla de prompt, guardrail nuevo | `foundation/` |
| Retrieval, chunking, vector store, embeddings | `generation/rag/` |
| Rol de agente o paso de orquestación ACB | `generation/agentic/` |
| Estrategia de caché | `generation/cag/` |
| Lógica de memoria conversacional | `generation/conversation/` |
| Endpoint HTTP | `api/` (fino) + factory en `dependencies.py` |
| Composición entre capas | método en `EstimationService` (`domain/`), **nunca** import cruzado |
| Fuente de datos / parser / limpieza offline | `ingestion/` |

## 9. Contratos públicos que NO se rompen

- **Rutas HTTP**: `/api/v1/estimate`, `/sessions/*`, `/api/v1/ingestion/*`,
  `/api/v1/embeddings/ingest`, `/api/v1/search`, `/api/v1/config/models`. El cliente Node
  (`cliente-web`) y Streamlit dependen de ellas y de la forma JSON de las respuestas.
- **`EstimationResult`** (`domain/schemas/estimations.py`): el orden de campos importa para
  Instructor (`phases` antes que `total_cost_eur` / `total_duration_weeks`) y los dos
  `model_validator` (`phases_sum_matches_total`, `low_confidence_requires_out_of_scope_prefix`)
  son las reglas de negocio que disparan el re-prompt.
- **`frontend/` no importa `app.*`**: su única frontera es HTTP. Lo verifica
  `tests/test_frontend_decoupling.py`.

## 10. Roadmap

- Conectar RAG al camino de petición (Query Understanding + retriever filtrado + context
  assembler) como pasos del conductor.
- Persistir el rollup del presupuesto (`project_summary`, `total_estimated_hours`,
  `component_count`) para que la generación citada no pierda los totales del padre.
- Si se quiere capas puras, sustituir el import perezoso de `app.dependencies` en los
  conductores por inyección explícita de un contexto de runtime.

## Apéndice — Mapa de migración (viejo → nuevo)

| Antes | Ahora |
|---|---|
| `app/services/estimation.py` | `app/domain/estimation_service.py` |
| `app/services/streaming.py` | `app/domain/streaming_service.py` |
| `app/services/llm_wrapper.py` | `app/foundation/llm/wrapper.py` |
| `app/services/runtime_config.py` | `app/foundation/llm/runtime_config.py` |
| `app/services/errors.py` | `app/foundation/llm/errors.py` |
| `app/services/cache.py` | `app/foundation/cache/backends.py` + `app/generation/cag/exact.py` |
| `app/cache/semantic.py` | `app/generation/cag/semantic.py` |
| `app/services/boss.py` | `app/generation/agentic/boss.py` |
| `app/services/critic.py` | `app/generation/agentic/critic.py` |
| `app/prompts/*` | `app/foundation/prompts/*` |
| `app/guardrails/*` | `app/foundation/guardrails/*` |
| `app/attachments/*` | `app/foundation/attachments/*` |
| `app/persistence/*` | `app/foundation/persistence/*` |
| `app/logging_config.py` | `app/foundation/observability/logging.py` |
| `app/schemas/*` | `app/domain/schemas/*` |
| `app/sessions/*` | `app/generation/conversation/*` |
| `app/embedding_pipeline/base.py` | `app/generation/rag/chunking/base.py` |
| `app/embedding_pipeline/structural.py` | `app/generation/rag/chunking/structural.py` |
| `app/embedding_pipeline/strategies/*` | `app/generation/rag/chunking/strategies/*` |
| `app/embedding_pipeline/embedder.py` | `app/generation/rag/embedding/embedder.py` |
| `app/embedding_pipeline/analysis/*` | `app/generation/rag/analysis/*` |
| `app/embedding_pipeline/store/*` | `app/generation/rag/store/*` |
| `app/embedding_pipeline/{schemas,ingest_service,retriever}.py` | `app/generation/rag/*` |
| `app/embedding_pipeline/router.py` | `app/api/embeddings.py` |
| `app/routers/*` | `app/api/*` |
| `app/ingestion/ports.py` | `app/domain/ports/ingestion.py` |
