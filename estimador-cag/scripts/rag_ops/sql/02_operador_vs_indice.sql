-- ============================================================
-- RAG Ops · Sesión 08 · Paso 2
-- El antipatrón silencioso: operador de query vs operator class del índice
-- ============================================================
--
-- El índice está construido con vector_cosine_ops (operador <=>).
-- Si una query ordena por <-> (distancia L2), Postgres no puede usar el índice
-- y cae a sequential scan. No hay error ni warning: solo un plan distinto.
--
-- Para no copiar vectores a mano usamos el embedding de un chunk existente.
-- El top-1 debe ser ese mismo chunk, con distancia 0.

-- Caso alineado: coseno (<=>) contra vector_cosine_ops.
-- En el plan debe aparecer "Index Scan using chunks_embedding_hnsw_idx".
EXPLAIN ANALYZE
SELECT id, chunk_type
FROM chunks
ORDER BY embedding <=> (SELECT embedding FROM chunks ORDER BY id LIMIT 1)
LIMIT 5;

-- Caso desalineado: L2 (<->) contra un índice de coseno.
-- En el plan debe aparecer "Seq Scan on chunks" seguido de un Sort.
-- Compara el Execution Time con el caso anterior.
EXPLAIN ANALYZE
SELECT id, chunk_type
FROM chunks
ORDER BY embedding <-> (SELECT embedding FROM chunks ORDER BY id LIMIT 1)
LIMIT 5;

-- Diagnóstico por estadísticas: idx_scan sube con el caso alineado y no con
-- el desalineado. Un índice vectorial con idx_scan = 0 en producción es casi
-- siempre este antipatrón.
SELECT
    indexrelname AS indice,
    idx_scan AS scans,
    last_idx_scan AS ultimo_uso
FROM pg_stat_user_indexes
WHERE relname = 'chunks'
ORDER BY idx_scan DESC;

-- Variante con el vector real del benchmark. baseline_latency imprime el
-- literal de la primera query; sustitúyelo en el CAST.
--
-- EXPLAIN ANALYZE
-- SELECT id, chunk_type
-- FROM chunks
-- ORDER BY embedding <=> CAST('<literal_pgvector>' AS vector)
-- LIMIT 5;
