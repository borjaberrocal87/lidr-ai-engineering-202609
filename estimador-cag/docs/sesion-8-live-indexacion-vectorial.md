# Indexación vectorial y operación (Sesión 8 en vivo)

> Cómo pasar del sequential scan al índice HNSW, medir el compromiso
> recall/latencia, probar halfvec y operar la base vectorial sin romper la API.
> Todo el material vive en `scripts/rag_ops/` y en este documento.

## Qué añade esta sesión

La Sesión 8 previa dejó el store persistido en pgvector y la búsqueda semántica
funcionando **sin índice vectorial**. Esta sesión añade la capa de datos que
falta:

- índice HNSW sobre `chunks.embedding`;
- índice halfvec sobre `(embedding::halfvec(1536))` y búsqueda opt-in;
- benchmarks de latencia, `ef_search` y comparación de índices;
- auditoría y mantenimiento de índices;
- corpus sintético reproducible para probar a escala.

No cambia el contrato de `/search` ni la UI. El único cambio de aplicación es
`SEARCH_USE_HALFVEC`, que decide si la query usa el índice de media precisión.

## RAG Ops

`scripts/rag_ops/` es el CLI de operaciones. Todos los comandos de este
documento se ejecutan **desde `estimador-cag/`** y con el stack levantado:

```bash
docker compose up -d --build
docker compose exec api alembic upgrade head
```

| Script | Comando | Qué comprueba |
|---|---|---|
| `baseline_latency` | `docker compose exec api python -m scripts.rag_ops.baseline_latency` | Latencia SQL pura de las 5 consultas de referencia, sin contar el embedding. Es el baseline antes de HNSW y la comparación después. |
| `ef_search_sweep` | `docker compose exec api python -m scripts.rag_ops.ef_search_sweep` | Barre `hnsw.ef_search`, mide recall@k contra la verdad de fondo y marca el valor recomendado. |
| `index_compare` | `docker compose exec api python -m scripts.rag_ops.index_compare` | Compara índice vectorial completo vs halfvec: top-k, overlap y latencia. Verifica con `EXPLAIN` qué índice usa cada plan. |
| `index_status` | `docker compose exec api python -m scripts.rag_ops.index_status` | Tamaño, tipo, `idx_scan`, último uso y operator class de cada índice de `chunks`. |
| `corpus_seed` | `docker compose exec api python -m scripts.rag_ops.corpus_seed 100` | Inserta chunks sintéticos con embeddings reales. Con `100` sirve para mantenimiento; `30000` es el preflight opcional. |

`common.py` no es un script: concentra las consultas de referencia, el formateo
de tablas y las estadísticas de latencia.

### `baseline_latency`

```bash
docker compose exec api python -m scripts.rag_ops.baseline_latency
docker compose exec api python -m scripts.rag_ops.baseline_latency --runs 5
```

**Qué comprueba.** Aísla el coste del `SELECT` de k-NN: el embedding se paga una
vez fuera del cronómetro. Imprime media, desviación, mínimo y máximo por query,
y al final el literal pgvector de la primera consulta para pegarlo en psql.

**Qué debe verse.** Sin índice, todas las queries caen en sequential scan. Con
HNSW, la media baja de forma visible en un corpus grande. Si el corpus es de 60
chunks, la diferencia puede ser de milisegundos: usa el preflight de 30.000
chunks para que el efecto sea medible.

**Cuándo ejecutarlo.** Antes de crear el índice (baseline) y justo después
(comparación).

### `ef_search_sweep`

```bash
docker compose exec api python -m scripts.rag_ops.ef_search_sweep
```

**Qué comprueba.** Primero calcula la verdad de fondo con
`SET LOCAL enable_indexscan = off` (sequential scan exacto). Después, para cada
`ef_search` de `[10, 20, 40, 80, 120, 200]`, mide latencia y recall@5 frente a
esa verdad de fondo. La fila con ★ es el valor más pequeño que mantiene el
recall dentro de 0,005 del mejor observado (o ≥ 0,99).

**Qué debe verse.** Una curva de recall que se satura y una latencia que sigue
creciendo. Si todos los valores dan el mismo plan, el índice HNSW no existe o
la query no lo está usando.

**Cuándo ejecutarlo.** Después de `01_hnsw.sql`. El valor recomendado es el
número que se lleva al entregable.

### `index_compare`

```bash
docker compose exec api python -m scripts.rag_ops.index_compare
```

**Qué comprueba.** Ejecuta las mismas queries contra el índice completo y
contra el halfvec. Calcula el overlap del top-k y la latencia de cada uno.
Antes de medir, lanza `EXPLAIN (FORMAT JSON)` para confirmar qué índice usa
realmente cada plan.

**Qué debe verse.** Un overlap alto (idealmente 1,00) y un tamaño de índice
halfvec aproximadamente la mitad. Si `EXPLAIN` devuelve `—`, algún índice no
existe o el planner no lo está usando.

**Cuándo ejecutarlo.** Después de `01_hnsw.sql` y `03_halfvec.sql`.

### `index_status`

```bash
docker compose exec api python -m scripts.rag_ops.index_status
docker compose exec api python -m scripts.rag_ops.index_status --json
```

**Qué comprueba.** Estado de todos los índices de `chunks`: tipo (`btree`, `gin`,
`hnsw`), tamaño, número de scans, último uso y operator class. Avisa si no hay
índice vectorial o si un índice HNSW tiene `idx_scan = 0`.

**Qué debe verse.** Tras servir búsquedas, `idx_scan` del índice usado debe
crecer. Un índice vectorial con 0 scans es sospechoso: casi siempre hay un
desajuste operador/operator class.

**Cuándo ejecutarlo.** Antes y después de cada decisión de índice (crear, borrar,
reindexar).

### `corpus_seed`

```bash
# Demo de mantenimiento: 100 chunks
docker compose exec api python -m scripts.rag_ops.corpus_seed 100

# Preflight opcional: primero simula, luego ejecuta si asumes el coste
docker compose exec api python -m scripts.rag_ops.corpus_seed 30000 --dry-run
docker compose exec api python -m scripts.rag_ops.corpus_seed 30000
```

**Qué comprueba.** Genera textos variados a partir de plantillas, los embebe
con el modelo real del proyecto y los persiste bajo el documento
`synthetic::rag-ops` con `chunk_type = 'synthetic'`. Los embeddings son reales:
vectores aleatorios no comparten la geometría del corpus y harían inútil el
recall.

**Qué debe verse.** Progreso por lotes y un resumen con chunks, tokens y coste
estimado. `--dry-run` no llama a la API ni escribe en Postgres. `--reset` borra
el documento sintético previo y sus chunks antes de sembrar.

**Cuándo ejecutarlo.** `100` para la demo de `VACUUM`/`REINDEX`; `30000` antes
del baseline si quieres que el sequential scan sea medible.

## Flujo recomendado

```bash
# 0. Corpus real ingestado (idempotente)
docker compose exec api python scripts/query_examples.py

# 1. Baseline sin índice
docker compose exec api python -m scripts.rag_ops.baseline_latency

# 2. Crear HNSW
docker compose exec -T estimator-postgres psql -U estimator -d estimator \
  < scripts/rag_ops/sql/01_hnsw.sql

# 3. Repetir baseline y demostrar el antipatrón
docker compose exec api python -m scripts.rag_ops.baseline_latency
docker compose exec -T estimator-postgres psql -U estimator -d estimator \
  < scripts/rag_ops/sql/02_operador_vs_indice.sql

# 4. Elegir ef_search
docker compose exec api python -m scripts.rag_ops.ef_search_sweep

# 5. Crear halfvec y comparar
docker compose exec -T estimator-postgres psql -U estimator -d estimator \
  < scripts/rag_ops/sql/03_halfvec.sql
docker compose exec api python -m scripts.rag_ops.index_compare

# 6. Pasar la API a halfvec (solo si el índice existe)
#    SEARCH_USE_HALFVEC=true en .env
docker compose up -d --force-recreate api
docker compose exec api python -m scripts.rag_ops.index_status

# 7. Mantenimiento y monitorización
docker compose exec api python -m scripts.rag_ops.corpus_seed 100
docker compose exec -T estimator-postgres psql -U estimator -d estimator \
  < scripts/rag_ops/sql/05_mantenimiento.sql
docker compose exec -T estimator-postgres psql -U estimator -d estimator \
  < scripts/rag_ops/sql/04_monitorizacion.sql

# 8. Revertir el ensayo (conserva el preflight)
docker compose exec -T estimator-postgres psql -U estimator -d estimator \
  -v keep_synthetic=30000 < scripts/rag_ops/sql/99_reversion.sql
```

## SQL paso a paso

| Archivo | Qué comprueba | Comando |
|---|---|---|
| `01_hnsw.sql` | Estado previo sin índice vectorial, creación de HNSW (`m=16`, `ef_construction=128`, `vector_cosine_ops`) y tamaño final. | `docker compose exec -T estimator-postgres psql -U estimator -d estimator < scripts/rag_ops/sql/01_hnsw.sql` |
| `02_operador_vs_indice.sql` | Que `<=>` usa el índice y `<->` cae a seq scan sin error; diagnóstico por `idx_scan`. | `... < scripts/rag_ops/sql/02_operador_vs_indice.sql` |
| `03_halfvec.sql` | Que el índice de expresión existe, ocupa ~la mitad y responde a la query casteada. | `... < scripts/rag_ops/sql/03_halfvec.sql` |
| `04_monitorizacion.sql` | Uso de índices, filas vivas/muertas y operator class de cada índice vectorial. | `... < scripts/rag_ops/sql/04_monitorizacion.sql` |
| `05_mantenimiento.sql` | Efecto de `ANALYZE`, `VACUUM ANALYZE` y `REINDEX CONCURRENTLY`. | `... < scripts/rag_ops/sql/05_mantenimiento.sql` |
| `99_reversion.sql` | Vuelta al estado preflight: sin índices HNSW, sin sintéticos de la demo y con estadísticas limpias. | `... -v keep_synthetic=30000 < scripts/rag_ops/sql/99_reversion.sql` |

Los índices se llaman `chunks_embedding_hnsw_idx` (vector completo) y
`chunks_embedding_halfvec_idx` (media precisión).

## Halfvec en la API

`SEARCH_USE_HALFVEC` está apagado por defecto:

```env
SEARCH_USE_HALFVEC=false
```

Actívalo **solo después** de crear `chunks_embedding_halfvec_idx`. Con el flag
activo, `ChunkStore` castea la columna a `halfvec(1536)` en la distancia; sin el
índice, ese cast por fila puede empeorar el plan. El contrato de `/search` no
cambia: el modo se registra en el log `embedding.search.end` como `search_mode`.

Para volver al índice completo:

```env
SEARCH_USE_HALFVEC=false
```

y recrea el contenedor `api`.

## Preflight de 30.000 chunks

Este repositorio **documenta** el preflight, pero no lo ejecuta. Tiene coste real
de embeddings (aproximadamente **$0,03** con `text-embedding-3-small`) y tarda
**5-10 minutos** según la red y la API.

```bash
# 1. Simulación: cuenta chunks, tokens y coste sin llamar a OpenAI
docker compose exec api python -m scripts.rag_ops.corpus_seed 30000 --dry-run

# 2. Ejecución real (solo si asumes el coste)
docker compose exec api python -m scripts.rag_ops.corpus_seed 30000
```

Los 30.000 chunks se conservan al revertir para no volver a pagar los
embeddings. Para limpiarlos del todo:

```sql
DELETE FROM documents WHERE source_path = 'synthetic::rag-ops';
```

La FK `ON DELETE CASCADE` borra también sus chunks.

## Monitorización canónica

Guarda esta consulta a mano:

```sql
SELECT
    indexrelname AS indice,
    idx_scan AS scans,
    COALESCE(last_idx_scan::text, 'nunca') AS ultimo_uso,
    pg_size_pretty(pg_relation_size(indexrelid)) AS tamano
FROM pg_stat_user_indexes
WHERE relname = 'chunks'
ORDER BY idx_scan DESC;
```

Si un índice vectorial tiene `idx_scan = 0` después de servir búsquedas
semánticas, revisa el operador de la query y la operator class del índice antes
de concluir que el índice no sirve.

## Entregable del directo

Rellena esta tabla con los números de **tu** barrido y justifica la decisión:

| `ef_search` | recall@5 | latencia media (ms) | p95 (ms) | ¿Elegido? |
|---|---|---|---|---|
| 10 | | | | |
| 20 | | | | |
| 40 | | | | |
| 80 | | | | |
| 120 | | | | |
| 200 | | | | |

La decisión debe responder: qué recall ganas, qué latencia pagas y por qué ese
valor es el mejor compromiso para el corpus y el presupuesto de latencia del
producto.

## Troubleshooting

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `idx_scan = 0` en el HNSW | Query con operador distinto a la operator class | Ejecuta `02_operador_vs_indice.sql` y compara `<=>` con `<->`. |
| `index_compare` muestra `—` | Falta uno de los índices | Ejecuta `01_hnsw.sql` y `03_halfvec.sql`. |
| Halfvec más lento que vector | Flag activo sin índice halfvec | Deja `SEARCH_USE_HALFVEC=false` o crea el índice. |
| `corpus_seed` dice que el documento ya existe | Ejecución previa del mismo `source_path` | Usa `--reset` o cambia `--source-path`. |
| `99_reversion.sql` falla con meta-comandos | Se ejecutó con un cliente que no es psql | Ejecútalo con `psql`, como indican los comandos. |
| `no hay cliente de embeddings configurado` | Falta `OPEN_AI_KEY` o endpoint propio | Configura `.env` y recrea `api`. |
