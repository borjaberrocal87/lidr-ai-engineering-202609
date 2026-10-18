# Sanity check de embeddings — Sesión 07

Comprobación mínima de que el pipeline de embeddings produce vectores que discriminan entre
textos semánticamente cercanos y lejanos.

## Entorno de ejecución

- Fecha: 2026-10-17
- Rama: `feature/pre-session-07`
- Modelo: `text-embedding-3-small` (OpenAI, 1536 dims, dimensión por defecto)
- Tokenizer del conteo: `tiktoken` (mismo modelo)
- Similitud: coseno calculado a mano con `math.sumprod` y `math.hypot` (stdlib, sin numpy)
- Ejecución: `scripts/compare.py`, dos textos por pareja

## Resultados

| Pareja | Texto A | Texto B | Coseno | Lectura |
|--------|---------|---------|--------|---------|
| A — cercanos | "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" | "Authorization service using JSON Web Tokens for a banking application" | **0.5957** | alta, pero justo por debajo del 0.6 orientativo |
| B — no relacionados | "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" | "Database migration from MySQL to PostgreSQL with zero downtime" | **0.1920** | baja, como se esperaba (< 0.4) |
| C — genéricos | "Backend services" | "API development" | **0.5407** | sin expectativa fija; resultado llamativo |

**Resumen:** A 0.5957 · B 0.1920 · C 0.5407.

## Comentario

La pareja B confirma que el modelo separa dos temas de ingeniería sin relación (0.19). La pareja A
es la interesante: 0.5957 se queda **a las puertas del 0.6** aunque ambos textos describen el mismo
concepto con vocabulario superficial casi disjunto; el umbral es una convención operativa, no una
frontera física. La pareja C es la más reveladora: dos frases cortas y vagas puntúan casi como A
porque aportan muy poca señal discriminativa.

## Implicación para el chunker

Si dos frases genéricas se parecen tanto, un componente "Authentication backend" sin contexto podría
acercarse a cualquier otro componente del corpus. Por eso `JSONStructuralChunker` antepone proyecto,
sector, año y tecnología principal: el vector conserva a quién pertenece el chunk. Es el mismo
mecanismo de *contextual chunk headers* que discutimos en el directo.

## Cómo reproducirlo

```bash
# Fuera del contenedor (con el .env cargado):
uv run python scripts/compare.py \
  --text-a "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" \
  --text-b "Authorization service using JSON Web Tokens for a banking application"

# Dentro del contenedor:
docker compose exec api python scripts/compare.py \
  --text-a "OAuth 2.0 authentication backend with JWT tokens for fintech mobile app" \
  --text-b "Authorization service using JSON Web Tokens for a banking application"
```

Repetir con las parejas B y C para reconstruir la tabla.

## Límites

- Es un sanity check, no una evaluación de retrieval (sin recall@k, NDCG ni golden set).
- Los umbrales 0.6 / 0.4 son orientativos y dependen del modelo y del fraseo.
- Los valores son deterministas para este modelo y estos textos. Si cambiamos `EMBEDDING_MODEL`,
  cambian la dimensión y los números, y dejan de ser comparables con esta tabla.
- Los vectores no se persisten todavía.
