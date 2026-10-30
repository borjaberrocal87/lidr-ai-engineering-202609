-- ============================================================
-- RAG Ops · Sesión 08 · Paso 1
-- Construcción del índice HNSW sobre la columna vectorial completa
-- ============================================================
--
-- Ejecutar desde la raíz de estimador-cag/:
--
--   docker compose exec -T estimator-postgres psql -U estimator -d estimator \
--     < scripts/rag_ops/sql/01_hnsw.sql
--
-- O abrir psql interactivo y pegar los bloques uno a uno:
--
--   docker compose exec estimator-postgres psql -U estimator -d estimator
--
-- Decisiones del índice:
--   · m = 16              conexiones por nodo; equilibrio habitual para 1536 dims.
--   · ef_construction=128 lista de candidatos al construir; más calidad de grafo
--                         que el default 64, a cambio de un build más lento.
--   · vector_cosine_ops   operator class de coseno. DEBE coincidir con el
--                         operador <=> de las queries. Si no coincide, Postgres
--                         ignora el índice sin error ni warning.

-- Estado de partida: solo deberían verse los índices relacionales.
SELECT
    indexrelname AS indice,
    pg_size_pretty(pg_relation_size(indexrelid)) AS tamano
FROM pg_stat_user_indexes
WHERE relname = 'chunks'
ORDER BY indexrelname;

-- Construcción del HNSW.
CREATE INDEX chunks_embedding_hnsw_idx
ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 128);

-- Verificación: el índice existe y ya ocupa espacio.
SELECT
    indexrelname AS indice,
    pg_size_pretty(pg_relation_size(indexrelid)) AS tamano
FROM pg_stat_user_indexes
WHERE relname = 'chunks'
ORDER BY pg_relation_size(indexrelid) DESC;

-- Siguiente paso: 02_operador_vs_indice.sql para comprobar que el planner
-- realmente lo usa con el operador correcto.
