# Justificación del diseño de persistencia vectorial (Sesión 8)

> Cuatro decisiones deliberadas detrás de `documents`/`chunks` en PostgreSQL +
> pgvector. El detalle técnico (esquema, migración, contratos y operación) vive
> en [`specs/sesion-8-persistencia-pgvector-busqueda-semantica.md`](../specs/sesion-8-persistencia-pgvector-busqueda-semantica.md).

## (a) Dos tablas y no una

Un presupuesto produce N chunks: es un uno-a-muchos real, no una tabla ancha. Si
todo viviera en `chunks`, la metadata del documento (`source_path`,
`document_type`, `ingested_at`) se repetiría en cada fila y nada impediría que
dos chunks del mismo presupuesto discreparan sobre su procedencia. Con
`documents` + `chunks` y `ON DELETE CASCADE`, borrar un presupuesto elimina sus
chunks automáticamente y la procedencia vive en un único sitio. El coste es un
`JOIN` cuando se necesita; el beneficio es integridad referencial.

## (b) `metadata` como JSONB y no columnas

Hay dos clases de metadata. La **estable** —tipo de documento, tipo de chunk,
fechas— se consulta, se indexa y se tipa: columna. La **variable** —sector,
tecnologías, horas, lo que el chunker decida añadir mañana— no justifica una
migración por cada clave nueva: JSONB. El índice GIN permite consultar claves
arbitrarias (`metadata->>'sector'`) sin tocar el esquema. La regla de promoción
es simple: cuando un campo se convierte en filtro caliente, se promociona a
columna tipada. En esta sesión no hay filtros; el GIN es la puerta que abre el
directo.

## (c) `cosine_distance` y no L2 ni inner product

Con embeddings normalizados de OpenAI, los tres rankings son equivalentes. La
elección no es matemática, es operativa: coseno es la convención de la
literatura RAG y, sobre todo, es la operator class (`vector_cosine_ops`) con la
que se construye el índice HNSW del directo. Si la query usa `<=>` y el índice
está en `vector_l2_ops`, PostgreSQL **ignora el índice en silencio** y vuelve al
sequential scan. Alinear operador y operator class desde el principio evita un
bug de rendimiento invisible.

## (d) Sin índice vectorial todavía

Con 17 presupuestos y 60 chunks, un sequential scan responde en pocos cientos de
milisegundos. Añadir HNSW ahora escondería el baseline: no sabríamos cuánto
mejora ni cuánto cuesta mantenerlo. El ejercicio mide `EXPLAIN ANALYZE` sin
índice; el directo añade el índice y compara. La disciplina de no adelantarse es
parte del valor.

**Fuera de alcance:** índices vectoriales, filtros por metadata, búsqueda
híbrida y tuning de PostgreSQL. Todo eso se construye en el directo.
