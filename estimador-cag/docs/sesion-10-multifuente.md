# Sesión 10 — Recuperación multifuente

Esta sesión amplía la recuperación de la S10 pre-work (híbrida + reranking) con
cuatro piezas que atacan fallos distintos del corpus real:

1. **Multi-índice**: presupuestos, transcripciones y documentación técnica viven
   en tablas separadas, con su propio esquema de metadata y su propio ciclo de
   vida.
2. **Enrutado de fuentes**: no buscar todo siempre; decidir qué fuentes pueden
   responder la consulta.
3. **Plan de consulta**: expandir (misma intención, varias formulaciones) o
   descomponer (varios temas, una sub-consulta por tema).
4. **Vigencia temporal**: la recencia desempata sin borrar la historia.

Además, el flujo de estimación cambia: la estructura se genera **sin retrieval**
y las horas se derivan **por tarea** contra el corpus histórico. La persona
revisa la estructura y valida las horas; el LLM no inventa cifras en el paso de
estructura.

## Arquitectura

```text
                         ┌────────────────────────────┐
consulta ──► plan de consulta │ expansión / descomposición │
                         └─────────────┬──────────────┘
                                       ▼
                         ┌────────────────────────────┐
                         │ enrutado de fuentes        │
                         │ explícito → reglas → LLM   │
                         └─────────────┬──────────────┘
                                       ▼
             ┌─────────────────────────┴─────────────────────────┐
             ▼                         ▼                         ▼
      budget_chunks            transcript_chunks         technical_doc_chunks
      (híbrida + filtros)      (híbrida + filtros)       (híbrida + filtros)
             └─────────────────────────┬─────────────────────────┘
                                       ▼
                         fusión diferenciada (RRF / round-robin)
                                       ▼
                         reranking (cross-encoder) → vigencia → top-k
```

## Catálogo de fuentes

`data/catalog/collections.yaml` **es el contrato**. Por cada fuente declara:

- `table`: tabla ORM que la respalda (`budget_chunks`, `transcript_chunks`,
  `technical_doc_chunks`).
- `chunk_type`: etiqueta estampada al ingerir.
- `source_id_key`: clave JSONB del documento padre (`budget_id`,
  `transcript_id`, `doc_id`).
- `date`: cómo leer la fecha (`year` o `iso`) para la vigencia.
- `filters`: ejes duros que entiende (`sectors`, `technologies`, `min_date`,
  `version`).
- `rules`: vocabulario determinista (ES + EN) para el nivel 1 del enrutado.

Añadir una fuente es editar el YAML y, si hace falta, registrar su modelo ORM en
`app/generation/rag/retrieval/sources.py`. El pipeline no cambia.

## Etapas

### Plan de consulta

`query_plan.py` elige por heurística:

- **Directa**: consultas cortas (< 6 palabras).
- **Expansión**: consultas medias; varias formulaciones de la misma intención,
  fusionadas por consenso (RRF).
- **Descomposición**: consultas largas o multitema; una sub-consulta por tema,
  fusionadas por cobertura (round-robin).

Ante fallo del LLM, degrada a consulta directa. La recuperación nunca depende de
que el transformador acierte.

### Enrutado

`routing.py` resuelve en cascada: explícito → reglas → clasificador LLM →
búsqueda en todas las fuentes. La ambigüedad se modela como **varios destinos**,
no como un umbral de confianza. El clasificador emite 1–3 fuentes y una razón.

### Filtros duros

Se embeben en la consulta SQL, antes del ranking. Evitan el post-filtrado
(silenciosamente caro y con cardinalidad impredecible). Cada fuente solo aplica
los ejes que declara en el catálogo.

### Búsqueda y fusión

- Dentro de una fuente: vectorial + léxica (tsvector/GIN) fusionadas con RRF.
- Entre sub-consultas: RRF para expansión, round-robin para descomposición.
- Entre fuentes: round-robin con clave `(collection, id)` (los ids solo son
  únicos dentro de una tabla).

### Reranking y vigencia

El cross-encoder reordena el pool. La vigencia se aplica **al final**, sobre
scores no negativos (los logits se mapean por sigmoide), con
`weight = 0.5 ** (edad_días / vida_media_días)`. Apagada por defecto: encenderla
solo con evidencia.

## Horas por tarea

La estructura se genera con `POST /api/v1/estimate/stages/structure`: módulos y
tareas sin horas, sin fuentes y sin citas. Después, por cada tarea:

1. se compone un texto `Module + Task + description`;
2. se embebe y se busca en `chunk_type='historical_task'`;
3. las horas son un **consenso ponderado por distancia** de los vecinos;
4. la fiabilidad combina cercanía y acuerdo (`1 - dispersión`);
5. sin vecino bajo el umbral → `has_match=false` y la UI marca la fila en rojo.

El resultado se revisa y se valida con horas + tarifa + coste.

## API

| Método | Ruta | Auth | Descripción |
| --- | --- | --- | --- |
| `POST` | `/api/v1/retrieval/multi-source` | `RETRIEVAL_API_KEY` | Recuperación multifuente con diagnóstico |
| `POST` | `/api/v1/estimate/stages/structure` | `ESTIMATE_API_KEY` | Estructura módulo→tarea sin horas |
| `POST` | `/api/v1/estimate/tasks/hours` | `ESTIMATE_API_KEY` | Consenso de horas por tarea |
| `PUT` | `/api/v1/config/retrieval` | — | Toggles y knobs en caliente |

`POST /api/v1/retrieval/multi-source` devuelve, además de los chunks, el nivel y
la razón del enrutado, la técnica, las sub-consultas y la cardinalidad por fuente
(un `0` es un vacío silencioso que hay que mirar).

## Configuración

```dotenv
RETRIEVAL_MULTI_SOURCE_ENABLED=true
RETRIEVAL_ROUTING_ENABLED=true
RETRIEVAL_QUERY_PLAN_ENABLED=true
RETRIEVAL_FRESHNESS_ENABLED=false
FRESHNESS_HALF_LIFE_DAYS=900
ROUTER_MODEL=gpt-4o-mini
QUERY_PLAN_MODEL=gpt-4o-mini
QUERY_MAX_SUBQUERIES=4
ROUTER_MAX_TARGETS=3
TASK_HOURS_TOP_K=5
TASK_HOURS_DISTANCE_THRESHOLD=0.45
```

`GET/PUT /api/v1/config/retrieval` permite cambiar en caliente `routing_enabled`,
`query_plan_enabled`, `freshness_enabled`, `task_hours_top_k` y
`task_hours_distance_threshold`.

## Puesta en marcha

```bash
# 1. Migrar el esquema
docker compose run --rm api alembic upgrade head

# 2. Ingerir presupuestos
docker compose run --rm api python scripts/query_examples.py

# 3. Ingerir transcripciones y documentación técnica
docker compose run --rm api python scripts/build_multi_index_corpus.py

# 4. Generar e ingerir el corpus de tareas
docker compose run --rm api python scripts/build_task_corpus.py --ingest

# 5. Evaluar (smoke barato: 1 consulta, 1 configuración, 1 repetición)
docker compose run --rm api python scripts/eval_retrieval_s10.py \
  --queries 1 --configs A --runs 1

# 5b. Barrido completo A-H (consume LLM en F/G/H)
docker compose run --rm api python scripts/eval_retrieval_s10.py
```

## Matriz de decisión

| Técnica | Qué fallo ataca | Coste | Cuándo usarla | Cuándo no |
| --- | --- | --- | --- | --- |
| Híbrida (FTS + RRF) | Vocabulario exacto que el embedding diluye | Bajo | Identificadores, acrónimos, nombres de producto | Consultas puramente semánticas |
| Reranking | Falsos positivos del recall | Medio (cross-encoder) | Precisión en el top-5 | Latencia muy ajustada |
| Enrutado | La fuente dominante inunda el top-k | Bajo/medio | Corpus heterogéneo | Una sola fuente |
| Expansión | Lotería de la formulación | Medio (LLM) | Consultas cortas y ambiguas | Consultas ya específicas |
| Descomposición | Embedding promediado de varios temas | Medio (LLM) | Consultas multitema | Consultas de un solo tema |
| Vigencia | Evidencia obsoleta | Bajo | El tiempo cambia la respuesta | Datos atemporales |

La regla es la misma que en el resto del proyecto: **cada etapa cara debe ganarse
su sitio con precisión medida**, no con una intuición.

## Límites conocidos

- El corpus de ejemplo es sintético; los números de la matriz deben rellenarse
  con la evaluación real de cada entorno.
- La configuración de FTS es `english` porque el corpus canónico está en inglés.
  Un corpus en español exige cambiar la migración y la consulta léxica.
- El enrutado por reglas es deliberadamente simple: cubre vocabulario, no
  intención compleja; para eso está el clasificador.
- La vigencia no sustituye a un filtro duro de fecha cuando la normativa exige
  excluir evidencia antigua.
