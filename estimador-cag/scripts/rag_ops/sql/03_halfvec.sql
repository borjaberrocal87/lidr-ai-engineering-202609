-- ============================================================
-- RAG Ops · Sesión 08 · Paso 3
-- Índice halfvec en paralelo al índice vectorial completo
-- ============================================================
--
-- halfvec guarda cada componente en 16 bits en vez de 32. La columna
-- embedding NO cambia: sigue siendo vector(1536). Lo que cambia es el índice,
-- que se construye sobre la expresión (embedding::halfvec(1536)).
--
-- Consecuencia importante: este índice solo sirve a queries que ordenen por
-- ESA MISMA expresión casteada. Una query con `embedding <=> ...` sin cast
-- usará el índice completo, no el halfvec.

-- Referencia de tamaño del índice completo.
SELECT pg_size_pretty(pg_relation_size('chunks_embedding_hnsw_idx')) AS vector_hnsw;

-- Índice de expresión. El doble paréntesis es obligatorio.
CREATE INDEX chunks_embedding_halfvec_idx
ON chunks
USING hnsw ((embedding::halfvec(1536)) halfvec_cosine_ops)
WITH (m = 16, ef_construction = 128);

-- Comparación directa de tamaños.
SELECT
    indexrelname AS indice,
    pg_size_pretty(pg_relation_size(indexrelid)) AS tamano
FROM pg_stat_user_indexes
WHERE relname = 'chunks'
  AND indexrelname LIKE 'chunks_embedding%'
ORDER BY pg_relation_size(indexrelid) DESC;

-- Verificación de que el planner usa el índice halfvec con la expresión
-- casteada. Debe aparecer "Index Scan using chunks_embedding_halfvec_idx".
EXPLAIN ANALYZE
SELECT id, chunk_type
FROM chunks
ORDER BY (embedding::halfvec(1536)) <=> (
    SELECT embedding::halfvec(1536)
    FROM chunks
    ORDER BY id
    LIMIT 1
)
LIMIT 5;

-- Para que la API use este índice, activa SEARCH_USE_HALFVEC=true en .env y
-- recrea el contenedor. Sin el índice, deja el flag en false.
