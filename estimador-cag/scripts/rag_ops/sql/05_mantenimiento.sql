-- ============================================================
-- RAG Ops · Sesión 08 · Paso 5
-- Ciclo de mantenimiento: ANALYZE, VACUUM y REINDEX CONCURRENTLY
-- ============================================================
--
-- Contexto: tras insertar chunks sintéticos con corpus_seed, las estadísticas
-- del planner quedan desfasadas. Este ciclo es el que mantiene sana una base
-- vectorial en producción.
--
-- OJO: VACUUM y REINDEX ... CONCURRENTLY no pueden ejecutarse dentro de una
-- transacción. En psql interactivo (autocommit) no hay problema; si pegas el
-- archivo completo, psql ejecuta sentencia a sentencia.

-- Estado previo.
SELECT
    n_live_tup AS filas_vivas,
    n_dead_tup AS filas_muertas,
    last_analyze AS ultimo_analyze,
    last_vacuum AS ultimo_vacuum
FROM pg_stat_user_tables
WHERE relname = 'chunks';

-- 1. ANALYZE: recalcula las estadísticas del planner. Rápido y no bloqueante.
--    Hazlo siempre después de una carga grande.
ANALYZE chunks;

SELECT last_analyze AS ultimo_analyze
FROM pg_stat_user_tables
WHERE relname = 'chunks';

-- 2. VACUUM ANALYZE: recupera el espacio de filas muertas y refresca
--    estadísticas. En una tabla de solo inserts habrá poco que recuperar; lo
--    importante es el hábito.
VACUUM ANALYZE chunks;

-- 3. REINDEX CONCURRENTLY: reconstruye el HNSW sin bloquear queries. Por
--    debajo crea un índice nuevo, espera a que esté listo y hace swap atómico.
--    Cuesta el doble de espacio temporal; en producción, ventana de bajo tráfico.
REINDEX INDEX CONCURRENTLY chunks_embedding_halfvec_idx;

-- Verificación posterior. En Postgres 16, REINDEX CONCURRENTLY conserva el
-- contador idx_scan del índice nuevo.
SELECT
    indexrelname AS indice,
    pg_size_pretty(pg_relation_size(indexrelid)) AS tamano,
    idx_scan AS scans
FROM pg_stat_user_indexes
WHERE relname = 'chunks'
ORDER BY pg_relation_size(indexrelid) DESC;

-- Cadencias orientativas:
--   ANALYZE   autovacuum lo dispara solo; manual tras cargas masivas.
--   VACUUM    semanal, o autovacuum bien configurado.
--   REINDEX   mensual, o ante degradación observada de recall/latencia.
