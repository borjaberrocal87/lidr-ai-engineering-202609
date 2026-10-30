-- ============================================================
-- RAG Ops · Sesión 08 · Reversión del ensayo
-- ============================================================
--
-- Devuelve la base de datos al estado de partida del directo:
--   · sin índices HNSW (completo y halfvec),
--   · sin los chunks sintéticos añadidos durante la demo,
--   · sin override global de hnsw.ef_search,
--   · con las estadísticas de uso reseteadas.
--
-- NO re-embebe nada: conserva los chunks del preflight para no pagar otra vez
-- los embeddings. Por defecto conserva 30.000 chunks sintéticos.
--
-- Ejecutar con psql (usa variables y meta-comandos):
--
--   docker compose exec -T estimator-postgres psql -U estimator -d estimator \
--     -v keep_synthetic=30000 < scripts/rag_ops/sql/99_reversion.sql
--
-- Idempotente: si no hay índices, no borra nada; si hay menos sintéticos que
-- keep_synthetic, tampoco elimina chunks.

\if :{?keep_synthetic}
\else
  \set keep_synthetic 30000
\endif

\echo '== Estado antes de revertir =='
SELECT
    (SELECT count(*) FROM chunks) AS total_chunks,
    (SELECT count(*) FROM chunks WHERE chunk_type = 'synthetic') AS synthetic_chunks;
SELECT indexrelname
FROM pg_stat_user_indexes
WHERE relname = 'chunks'
ORDER BY indexrelname;

-- 1. Borrar solo los sintéticos posteriores al preflight. Los del preflight
--    tienen los ids más bajos; si hay <= keep_synthetic, la subconsulta
--    devuelve NULL y no se borra nada.
DELETE FROM chunks
WHERE chunk_type = 'synthetic'
  AND id > (
      SELECT id
      FROM chunks
      WHERE chunk_type = 'synthetic'
      ORDER BY id
      OFFSET (:keep_synthetic - 1)
      LIMIT 1
  );

-- 2. Dropear los índices vectoriales del ensayo. Incluye índices INVALID que
--    hayan quedado tras un CONCURRENTLY abortado.
DROP INDEX IF EXISTS chunks_embedding_hnsw_idx;
DROP INDEX IF EXISTS chunks_embedding_halfvec_idx;

-- 3. Revertir el override global de ef_search, si se fijó con ALTER DATABASE.
--    Los SET / SET LOCAL de sesión no persisten y no hace falta tocarlos.
ALTER DATABASE estimator RESET hnsw.ef_search;

-- 4. Resetear estadísticas para que la demo del antipatrón arranque limpia.
SELECT pg_stat_reset();
ANALYZE chunks;

\echo '== Estado después de revertir =='
SELECT
    (SELECT count(*) FROM chunks) AS total_chunks,
    (SELECT count(*) FROM chunks WHERE chunk_type = 'synthetic') AS synthetic_chunks,
    (SELECT count(*) FROM documents WHERE document_type = 'synthetic_test') AS synthetic_docs;
SELECT indexrelname
FROM pg_stat_user_indexes
WHERE relname = 'chunks'
ORDER BY indexrelname;
