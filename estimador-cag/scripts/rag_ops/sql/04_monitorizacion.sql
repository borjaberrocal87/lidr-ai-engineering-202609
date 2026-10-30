-- ============================================================
-- RAG Ops · Sesión 08 · Paso 4
-- Monitorización del store vectorial
-- ============================================================
--
-- Estas consultas son las que se llevan a producción. La primera es la
-- canónica para responder "¿por qué va lento?": tamaño, uso y último scan de
-- cada índice de chunks.

-- Query canónica de índices.
SELECT
    indexrelname AS indice,
    idx_scan AS scans,
    COALESCE(last_idx_scan::text, 'nunca') AS ultimo_uso,
    pg_size_pretty(pg_relation_size(indexrelid)) AS tamano
FROM pg_stat_user_indexes
WHERE relname = 'chunks'
ORDER BY idx_scan DESC;

-- Actividad de la tabla: filas vivas, filas muertas y frescura de estadísticas.
SELECT
    relname AS tabla,
    n_live_tup AS filas_vivas,
    n_dead_tup AS filas_muertas,
    last_analyze AS ultimo_analyze,
    last_autoanalyze AS ultimo_autoanalyze,
    last_vacuum AS ultimo_vacuum
FROM pg_stat_user_tables
WHERE relname = 'chunks';

-- Operator class de cada índice vectorial. Sirve para confirmar que el índice
-- de coseno no está recibiendo queries de L2.
SELECT
    i.indexrelname AS indice,
    op.opcname AS operator_class
FROM pg_stat_user_indexes i
JOIN pg_index pgi ON pgi.indexrelid = i.indexrelid
JOIN pg_opclass op ON op.oid = ANY (pgi.indclass::oid[])
WHERE i.relname = 'chunks'
  AND (op.opcname LIKE 'vector%' OR op.opcname LIKE 'halfvec%')
ORDER BY i.indexrelname, op.opcname;

-- Si un índice vectorial tiene idx_scan = 0 después de servir búsquedas:
--   1. Revisa la operator class de esta consulta.
--   2. Revisa el operador de la query (02_operador_vs_indice.sql).
--   3. Solo entonces plantea que el índice no sea necesario.
--
-- Complementos fuera de psql:
--   · pg_stat_statements para ranking de queries por tiempo acumulado.
--   · El evento estructurado embedding.search.end del servicio, que incluye
--     search_mode y search_time_ms.
