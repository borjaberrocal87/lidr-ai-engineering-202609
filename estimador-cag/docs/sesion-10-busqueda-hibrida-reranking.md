# Sesión 10 — Búsqueda híbrida y reranking

## Objetivo y alcance

Esta sesión ataca el falso positivo semántico: un presupuesto puede estar cerca en el espacio
vectorial y no ser relevante para la consulta. Se añaden dos técnicas y, sobre todo, se miden:

1. **Búsqueda híbrida**: rama vectorial + rama léxica de PostgreSQL, fusionadas con Reciprocal
   Rank Fusion (RRF).
2. **Reranking**: un cross-encoder reordena un recall amplio (top-50) y deja los mejores (top-5).

La comparación es entre cuatro configuraciones sobre un golden set de cinco consultas:

| Configuración | Búsqueda | Reranking |
|---|---|---|
| A | Vectorial | No |
| B | Híbrida | No |
| C | Vectorial | Sí |
| D | Híbrida | Sí |

**Fuera de alcance**: expansión de consultas, routing multi-índice, filtros de metadata nuevos y
cambios en la UI. Los filtros estructurales de la Sesión 09 (sector, año, tipo de chunk) se
reutilizan en ambas ramas.

## Diseño

### Rama léxica

La migración `0003_session10_fts` añade a `chunks` una columna generada:

```sql
content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
```

con su índice GIN `ix_chunks_content_tsv`. La configuración es `english` porque el corpus canónico
de este repo (`data/budgets_sample.json`, `data/task_corpus.json`) está en inglés, aunque el
enunciado diga que está en español. Si el corpus cambia de idioma, hay que cambiar `english` por
`spanish` en la migración y en la consulta, y recrear la columna.

La rama léxica usa `websearch_to_tsquery` con **OR entre términos** (`_or_tsquery_text`) y
`ts_rank_cd` para ordenar. Esto es deliberado: `plainto_tsquery` construye un AND de todos los
lexemas y, con descripciones de proyecto de 20 palabras, devuelve cero filas. La precisión la
resuelven RRF y el cross-encoder; el recall léxico solo tiene que aportar candidatos.

### Fusión RRF

`reciprocal_rank_fusion` combina las posiciones de las dos ramas:

```text
score(doc) = sum(1 / (k + rank))   con k = 60
```

No se normalizan los scores porque la distancia coseno (menor = mejor) y `ts_rank_cd` (mayor =
mejor) no son comparables. RRF solo usa el orden.

### Reranker

`CrossEncoderReranker` envuelve
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. El modelo se carga de forma perezosa (importar el
módulo no importa torch) y se ejecuta en un hilo (`asyncio.to_thread`) para no bloquear el event
loop. El pipeline recupera `recall_k=50` candidatos y el cross-encoder los reduce a
`rerank_top_n=5`.

### Toggles

- `.env`: `RETRIEVAL_SEARCH_MODE`, `RERANKER_ENABLED`, `RERANKER_MODEL`,
  `RETRIEVAL_RECALL_TOP_K`, `RERANK_TOP_N`, `RRF_K`.
- Petición: `RetrievalQuery.search_mode` y `RetrievalQuery.rerank`.
- Runtime: `GET/PUT /api/v1/config/retrieval`, respaldado por el hash Redis
  `estimator:runtime_retrieval`.

Precedencia: **petición → runtime → settings**.

## Golden set y metodología

`evals/golden_retrieval.json` contiene cinco consultas representativas, anotadas a mano por
presupuesto padre (`budget_id`). Se evalúan solo chunks `budget_component` para que la métrica sea
reproducible aunque el corpus task-level también esté ingerido.

Metodología:

- Embeddings de las cinco consultas calculados **una sola vez**, fuera de la latencia.
- Warm-up del reranker descartado (la primera carga descarga pesos).
- Tres ejecuciones medidas por consulta y configuración.
- Umbral de distancia permisivo (`2.0`) para comparar ranking, no el soft-fail.
- `precision@5 = relevantes_en_top5 / 5`; un chunk es relevante si su `budget_id` está en la
  anotación.
- La latencia mide retrieval + fusión + rerank, no el round-trip de embeddings.

## Resultados

Entorno: Docker Compose en un Mac, CPU, corpus de 60 chunks `budget_component`. La latencia
absoluta depende de la máquina; lo relevante es la comparación relativa.

| Config | Search | Reranking | Precision@5 | Latency (ms) |
|---|---|---|---|---|
| A | Vector | No | 0.92 | 1.9 |
| B | Hybrid | No | 0.88 | 4.0 |
| C | Vector | Yes | 0.92 | 1841.5 |
| D | Hybrid | Yes | 0.92 | 2175.6 |

### Desglose por consulta

| Query | A | B | C | D |
|---|---|---|---|---|
| Q1 | 1.00 | 0.80 | 0.80 | 0.80 |
| Q2 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q3 | 0.80 | 0.80 | 0.80 | 0.80 |
| Q4 | 0.80 | 0.80 | 1.00 | 1.00 |
| Q5 | 1.00 | 1.00 | 1.00 | 1.00 |

Lectura:

- **A es el mejor baseline**: 0.92 de precisión con ~2 ms. El corpus es pequeño y el embedding ya
  resuelve bien la mayoría de consultas.
- **B no mejora a A** (0.88): la rama léxica mete candidatos que RRF coloca por delante de
  relevantes en Q1. El coste es pequeño (~2 ms extra), pero no compensa.
- **C y D empatan con A en precisión media (0.92)** y cuestan ~1.8–2.2 s por consulta. El reranker
  mejora Q4 (0.80 → 1.00) pero empeora Q1 (1.00 → 0.80); el neto es cero.
- La latencia del cross-encoder es ~1000× la del retrieval vectorial en esta máquina y con este
  modelo. En un caso de uso interactivo, ese coste solo se justifica si la precisión sube de forma
  clara y consistente.

## Evidencia

Gate del reranker:

```text
Loading reranker model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 ...
reranker.loaded  load_ms=17708
reranker.scored  pairs=2 score_ms=159
Query:          'e-commerce checkout and shopping cart platform'
Relevant doc:   score = 2.9526
Irrelevant doc: score = -9.0109
OK: reranker loaded and ranked the relevant document first.
```

Migración:

```text
version_num
------------------------
0003_session10_fts
```

Evaluación:

```bash
docker compose exec api python scripts/eval_retrieval_s10.py \
  | tee data/output/session10/retrieval_eval.txt
```

La salida completa está en `data/output/session10/retrieval_eval.txt`. El script también imprime
`lexical_hits` por consulta en los logs; con OR, las cinco consultas recuperan entre 11 y 34
candidatos léxicos.

## Conclusiones

**Configuración elegida para este proyecto: A (vectorial, sin reranking) como default.** Con el
corpus y el golden set actuales, el embedding ya alcanza 0.92 de precisión@5 y el reranking no
mejora la media; además cuesta ~1.8–2.2 s por consulta frente a ~2 ms. La ganancia de relevancia no
justifica la latencia en este caso de uso concreto. La búsqueda híbrida tampoco compensa por sí
sola: B baja a 0.88 porque la rama léxica introduce ruido que RRF no filtra.

Dicho esto, la conclusión no es “el reranking no sirve”: es “en este corpus pequeño, con cinco
consultas y este modelo, no se paga”. Si el corpus creciera, si las consultas fueran más ambiguas o
si el golden set midiera errores que el vector no ve (jerga exacta, códigos, tecnologías), el
reranker podría subir la precisión. Para producción, antes de activarlo por defecto habría que:
medir con un golden set más grande y difícil, usar un modelo más pequeño o GPU, cachear por query y
rerankear solo cuando el recall sea de baja confianza.

## Limitaciones

- El corpus es pequeño (60 chunks) y el golden set tiene cinco consultas; la precisión@5 está
  dominada por pocos casos.
- La latencia se midió en CPU dentro de Docker; no es extrapolable a un servidor con GPU.
- El modelo de reranking es multilingüe y pequeño; un modelo mayor podría mejorar la precisión a
  cambio de más latencia.
- `precision@5` con dos presupuestos relevantes tiene techo 0.4 en algunas consultas; el desglose
  por consulta es más informativo que la media sola.
- La rama léxica usa OR y no soporta frases entrecomilladas; es una decisión de recall, no de
  precisión.
- La imagen de Docker crece porque `sentence-transformers` arrastra `torch`. En Linux, el wheel por
  defecto incluye dependencias CUDA aunque se ejecute en CPU; para producción conviene usar el
  índice CPU de PyTorch.

## Operación

Verificación:

```bash
docker compose exec api python -m app.generation.rag.retrieval.verify_reranker
```

Configuración en caliente:

```bash
curl -s http://localhost:8000/api/v1/config/retrieval | jq
curl -s -X PUT http://localhost:8000/api/v1/config/retrieval \
  -H 'Content-Type: application/json' \
  -d '{"search_mode": "hybrid", "rerank": true}' | jq
curl -s -X PUT http://localhost:8000/api/v1/config/retrieval \
  -H 'Content-Type: application/json' \
  -d '{"search_mode": null, "rerank": null}' | jq
```

Evaluación:

```bash
uv run python scripts/eval_retrieval_s10.py
```

En este entorno, el camino recomendado es dejar `RETRIEVAL_SEARCH_MODE=vector` y
`RERANKER_ENABLED=false`; activar `hybrid`/`rerank` solo para experimentos o consultas concretas
donde se haya comprobado que ayudan.
