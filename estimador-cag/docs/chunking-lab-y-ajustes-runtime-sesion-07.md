# Chunking Lab y ajustes de modelos en runtime (Sesión 7)

> Cómo pasamos del chunker estructural del ejercicio a un laboratorio de
> comparación de ocho estrategias y a un cambio de modelos LLM en caliente,
> sin tocar `.env` ni recrear contenedores.

Este documento cubre las dos funcionalidades que la Sesión 7 incorpora al
proyecto:

1. **Chunking Lab**: comparar estrategias de chunking sobre el corpus de
   presupuestos, con estadísticos, coste y top-k por similitud.
2. **Ajustes runtime**: overrides de los modelos LLM en Redis, editables desde
   la UI y aplicables en la siguiente llamada.

La implementación adapta el material de referencia a nuestra arquitectura
plana (`app/embedding_pipeline`, `app/services`, `app/routers`) y a nuestro
cliente Node/Express. **No** se copia el refactor de capas de la referencia.

---

## 1. Chunking Lab

### 1.1 Estrategias

Todas implementan la interfaz `Chunker` (`app/embedding_pipeline/base.py`):

| Estrategia | API externa | Qué hace |
|---|---|---|
| `structural` | No | Un chunk por componente con cabecera del presupuesto padre. |
| `fixed_size` | No | Ventana fija de 512 tokens con 80 de solape. El baseline. |
| `recursive` | No | Divide por párrafo, línea, frase y palabra (LangChain). |
| `sentence_window` | No | Indexa frases y guarda ±2 frases en `metadata.window_text`. |
| `semantic` | OpenAI embeddings | Corta donde cae la similitud entre frases. Coste de ingesta extra. |
| `propositional` | OpenAI (LLM) | Una llamada por componente para extraer proposiciones atómicas. |
| `contextual_retrieval` | Anthropic | Añade un contexto generado a cada chunk (prompt caching). |
| `hierarchical` | No | Padres (presupuesto completo) e hijos (componentes). |

Las estrategias de pago son **opt-in** en la UI y muestran su coste estimado en
la respuesta (`ingestion_cost_usd`, `ingestion_seconds`).

### 1.2 Módulos

```
app/embedding_pipeline/
├── base.py                  # Chunker, count_tokens, emit_chunking_done
├── structural.py            # JSONStructuralChunker + helpers de render
├── chunker.py               # shim de compatibilidad (imports históricos)
├── strategies/
│   ├── fixed_size.py
│   ├── recursive.py
│   ├── sentence_window.py
│   ├── semantic.py
│   ├── propositional.py
│   ├── contextual_retrieval.py
│   └── hierarchical.py
├── analysis/
│   ├── similarity.py        # coseno y percentiles (stdlib)
│   └── comparison.py        # stats + top-k + memoización
├── schemas.py               # contratos de ingesta y chunking
├── embedder.py              # OpenAIEmbedder (batches, reintentos, dimensions)
└── router.py                # /api/v1/embeddings/{ingest,compare}
```

### 1.3 Endpoint de comparación

```
POST /api/v1/embeddings/compare
{
  "budgets": [ <Budget>, ... ],
  "queries": ["OAuth authentication", ...],
  "strategies": ["structural", "hierarchical"],
  "top_k": 3
}
```

Respuesta:

```json
{
  "stats_per_strategy": {
    "structural": {
      "strategy": "structural",
      "n_chunks": 60,
      "token_distribution": {"min": 22, "p50": 78, "p95": 140, "max": 190},
      "n_orphan_chunks": 0,
      "n_obese_chunks": 0,
      "ingestion_cost_usd": 0.0,
      "ingestion_seconds": 0.01
    }
  },
  "queries_per_strategy": {
    "structural": [
      {
        "strategy": "structural",
        "query": "OAuth authentication",
        "top_k": [
          {"chunk_id": "BUD-2024-001::AUTH-001", "cosine": 0.81, "text_preview": "..."}
        ]
      }
    ]
  }
}
```

Notas:

- Si `queries` está vacío, **no se necesitan embeddings** y la comparativa es
  solo de estadísticos. Si hay queries, el embedder es obligatorio.
- `strategies` vacío equivale a todas.
- `top_k` se limita a 1..10.
- Nada se persiste en el backend; la persistencia vive en el cliente Node.

### 1.4 CLI

```bash
# Estadísticos de las estrategias gratuitas (sin API key)
uv run python scripts/compare_chunkers.py \
  --strategies structural,fixed-size,recursive,sentence-window,hierarchical \
  --show-stats --show-cost

# Top-k de una query para dos estrategias (requiere OpenAI)
uv run python scripts/compare_chunkers.py \
  --strategies sentence-window,structural \
  --queries "OAuth authentication for fintech mobile app" \
  --show-top-k 3

# Comparar dimensiones de embedding
uv run python scripts/compare_chunkers.py --models small-1536,small-768

# Generar el informe Markdown
uv run python scripts/compare_chunkers.py --strategies all --queries all \
  --show-stats --show-cost --output app/embedding_pipeline/COMPARISON_REPORT.md
```

El informe con las estrategias gratuitas está versionado en
`app/embedding_pipeline/COMPARISON_REPORT.md`. El corpus vive en
`data/budgets_sample.json` (17 presupuestos, 60 componentes) y las queries de
ejemplo en `data/test_queries.json`. El cliente Node mantiene una
copia en `cliente-web/data/`; un test comprueba que ambas copias son idénticas.

### 1.5 UI: `/chunking-lab`

- `GET /chunking-lab`: historial de comparativas persistidas.
- `GET /chunking-lab/new`: formulario de estrategias, queries y top-k.
- `POST /chunking-lab`: ejecuta la comparativa y persiste el run.
- `GET /chunking-lab/:id`: estadísticos, barras de coste y top-k por query.

La persistencia usa la tabla `chunking_comparisons` (migración
`002_create_chunking_comparisons.sql`) en el Postgres del cliente. Cada run
guarda el payload completo (`response_payload` JSONB) y su duración; así se
pueden revisitar comparativas caras sin volver a pagarlas. Si no hay
`DATABASE_URL`, el cliente usa un repositorio en memoria.

El timeout de la llamada es de **600 s** (`CHUNKING_TIMEOUT_MS`), separado del
timeout general de estimaciones (60 s).

---

## 2. Ajustes runtime

### 2.1 Qué se puede cambiar

| Clave API | Campo `.env` | Herencia |
|---|---|---|
| `PRIMARY_MODEL` | `LLM_MODEL` | — |
| `FALLBACK_MODEL` | `LLM_FALLBACK_MODEL` | — |
| `CRITIC_MODEL` | `CRITIC_MODEL` | Vacío ⇒ `LLM_MODEL` |
| `METADATA_EXTRACTOR_MODEL` | `METADATA_EXTRACTOR_MODEL` | Vacío ⇒ `LLM_MODEL` |
| `COMPRESSION_MODEL` | `COMPRESSION_MODEL` | Vacío ⇒ `LLM_MODEL` |
| `PROPOSITIONAL_CHUNKER_MODEL` | `PROPOSITIONAL_CHUNKER_MODEL` | — |
| `CONTEXTUAL_CHUNKER_MODEL` | `CONTEXTUAL_CHUNKER_MODEL` | — |

`EMBEDDING_MODEL` queda fuera a propósito: cambiarlo invalidaría todos los
vectores almacenados.

### 2.2 API

```
GET /api/v1/config/models
→ {
    "models": {
      "PRIMARY_MODEL": {"effective": "gpt-4o-mini", "default": "gpt-4o-mini", "overridden": false},
      ...
    },
    "available_models": ["gpt-4o-mini", "gpt-4o", "claude-haiku-4-5", "claude-sonnet-4-5"],
    "embedding_model": "text-embedding-3-small",
    "embedding_model_note": "Solo lectura: ..."
  }

PUT /api/v1/config/models
{"models": {"PRIMARY_MODEL": "gpt-4o", "CRITIC_MODEL": null}}
→ mismo shape que el GET
```

- Actualización **parcial**: solo se tocan las claves presentes.
- `null` resetea al default de `.env`.
- Todo o nada: se valida el lote completo antes de escribir.
- 422 clave desconocida o modelo fuera del catálogo.
- 400 modelo válido pero sin API key del proveedor.
- 503 Redis no disponible al escribir.
- Las lecturas degradan a los defaults de `.env` si Redis cae.

### 2.3 Store Redis

`app/services/runtime_config.py` guarda los overrides en el hash
`estimator:runtime_config`. Redis es la fuente de verdad y se lee en cada
llamada, así que:

- el cambio aplica en la siguiente petición;
- sobrevive a reinicios de uvicorn/`--reload`;
- todos los workers ven el mismo valor.

El catálogo (`AVAILABLE_MODELS`) se filtra por las API keys realmente
configuradas. Los modelos custom solo aparecen si hay
`CUSTOM_LLM_BASE_URL` + `CUSTOM_LLM_API_KEY`.

### 2.4 Integración con el pipeline

- `LLMWrapper.primary_model` y `fallback_model` son properties que resuelven el
  override por llamada.
- Si hay un override de primario/fallback, `complete()` y `complete_stream()`
  usan LiteLLM directo (sin reconstruir el Router). El camino estructurado ya
  llamaba a LiteLLM directo, así que solo cambia el modelo/proveedor efectivo.
- `EstimationService` resuelve critic, metadata y compresión por llamada y usa
  el modelo efectivo en la clave de la caché exacta.
- La caché semántica incluye el modelo LLM en su bucket: un hit generado por un
  modelo nunca se sirve con otro activo.
- Las factorías de `propositional` y `contextual_retrieval` no usan `lru_cache`
  para que el override aplique en la siguiente comparativa.

### 2.5 UI: `/ai-settings`

- Un select por knob; la opción vacía significa “usar el default de `.env`”.
- Muestra el valor por defecto y una marca `override` cuando aplica.
- El modelo de embeddings se muestra como solo lectura.
- El modelo primario efectivo aparece como badge en la navbar (caché de 15 s,
  timeout de 3 s, oculto si la API no responde).
- Si la API o Redis fallan, Ajustes muestra el error y no bloquea el resto de
  la aplicación.

---

## 3. Variables de entorno nuevas

| Variable | Default | Descripción |
|---|---|---|
| `PROPOSITIONAL_CHUNKER_MODEL` | `gpt-4o-mini` | Modelo del chunker proposicional. |
| `CONTEXTUAL_CHUNKER_MODEL` | `claude-sonnet-4-5` | Modelo del chunker contextual. |
| `AVAILABLE_MODELS` | catálogo del código | Lista JSON de modelos seleccionables. |
| `CHUNKING_TIMEOUT_MS` | `600000` | Timeout del cliente Node para `/embeddings/compare`. |
| `CONFIG_TIMEOUT_MS` | `3000` | Timeout del badge/Ajustes. |
| `CONFIG_CACHE_TTL_MS` | `15000` | TTL de la caché del badge. |

---

## 4. Cómo probarlo

### Backend

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy app frontend
```

Tests nuevos relevantes:

- `tests/test_chunking_strategies.py`: estrategias offline y con LLM falso.
- `tests/test_embedding_compare.py`: contrato del endpoint y modo sin embedder.
- `tests/test_runtime_config.py`: store Redis con fakeredis.
- `tests/test_config_endpoint.py`: validaciones y errores 400/422/503.
- `tests/test_llm_wrapper_runtime.py`: properties y camino directo.

### Cliente Node

```bash
cd cliente-web
npm test
npm run build:css
```

Tests nuevos: corpus sincronizado, `HttpEmbeddingsClient`, `HttpConfigClient`,
rutas de Chunking Lab y Ajustes.

### Demo end-to-end

```bash
docker compose up --build
```

1. Abre `http://localhost:3000/chunking-lab/new`.
2. Deja las estrategias gratuitas y ejecuta sin queries: no necesita API key.
3. Añade `structural` + `hierarchical` y una query: requiere `OPEN_AI_KEY` para
   el top-k.
4. Marca `propositional` o `contextual_retrieval` solo si tienes las claves y
   asumes el coste/tiempo.
5. Abre `http://localhost:3000/ai-settings`, cambia el modelo primario y guarda.
6. Lanza una estimación y comprueba en los logs `llm.structured.start` que el
   modelo efectivo es el nuevo.
7. Reinicia el contenedor `api` y verifica que el override sigue activo.
8. Vuelve a Ajustes y elige “Por defecto” para resetear.

---

## 5. Límites y troubleshooting

- **Sin retrieval real**: `retriever.py` y `store/` siguen reservados para la
  Sesión 8 (pgvector). Aquí solo se comparan chunks, no se persisten vectores.
- **Late chunking** se queda como concepto: requiere modelos con token-level
  embeddings que no usamos.
- **Estrategias de pago**: `semantic`, `propositional` y
  `contextual_retrieval` llaman a APIs externas. Si faltan claves, el endpoint
  devuelve 500 con un mensaje accionable.
- **NLTK**: `sentence_window` descarga `punkt`/`punkt_tab` de forma perezosa. En
  un entorno sin red, predescarga esos recursos en la imagen.
- **Streaming**: el camino SSE mantiene el comportamiento; si hay override, usa
  el modelo efectivo y no aplica fallback automático durante el stream.
- **Redis caído**: las lecturas de overrides caen a `.env`; las escrituras
  devuelven 503. El resto de la app sigue funcionando.
- **`AVAILABLE_MODELS`**: si lo defines en `.env`, debe ser JSON válido. Si no,
  pydantic-settings fallará al arrancar.
