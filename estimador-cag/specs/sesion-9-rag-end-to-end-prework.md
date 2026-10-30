# Spec — Proyecto 3: RAG end-to-end (pre-work de diagnóstico arquitectónico)

## Objetivo

La Sesión 08 cerró con un servicio IA que ya sabe hacer dos cosas: **vectorizar presupuestos
históricos** y **devolver los chunks más similares a una consulta**. El objetivo del proyecto, sin
embargo, es otro: recibir la transcripción de una reunión y devolver una **estimación fundamentada**
en presupuestos históricos. Entre lo que hay y lo que el sistema necesita sigue habiendo un hueco.

La Sesión 09 entra a llenar ese hueco con el flujo RAG completo (Query → Retrieval → Augmentation →
Generation). Antes de construir nada nuevo, este pre-work pide **mirar despacio lo que ya existe**:
ejecutar el sistema contra una transcripción realista, observar qué devuelve y razonar sobre los
gaps desde la evidencia, no desde la intuición.

Este ejercicio **no pide código nuevo**. Pide producir un único documento de diagnóstico
arquitectónico (`arquitectura-actual.md`) que:

1. Describa el estado actual del servicio IA tras las Sesiones 06–08.
2. Registre el comportamiento observable del sistema con una transcripción cruda.
3. Identifique cinco fallos concretos y verificables.
4. Proponga, a nivel de cajas y flechas, cómo debe evolucionar la arquitectura.

Al terminar tendrás:

- `arquitectura-actual.md` en la raíz de `estimador-cag/`, con las cuatro secciones obligatorias.
- Tres transcripciones de ejemplo en `examples/transcripts/` y una plantilla de entrega.
- Un script **cliente** `examples/trace_s09.py` que ejecuta el trace sin añadir comportamiento al
  servicio.
- Un mapa mental claro de tu propio sistema y preguntas concretas formuladas desde la evidencia.

> **Este pre-work no implementa la Sesión 09.** No se escribe reformulación de queries, ni reranking,
> ni un retriever nuevo, ni el módulo de generación. Eso es trabajo del directo y de las sesiones
> siguientes. Si te ves escribiendo Python más allá del script cliente del trace, has salido del
> alcance.

## Contexto del proyecto

El punto de partida es `main` con las Sesiones 06–08 cerradas. En este repo el servicio IA no se
llama `ingest/`, `embedding_pipeline/` y `storage/` de forma genérica: tiene nombres y fronteras
concretas. El diagnóstico debe describir **lo que hay de verdad**, no la arquitectura ideal del
enunciado.

### Capas existentes

- **Frontend / presentación**
  - `frontend/streamlit_app.py` — UI conversacional Streamlit; consume la API por HTTP a través de
    `frontend/client.py`, sin importar `app.*`.
  - `cliente-web/` — cliente de negocio Node/Express + EJS + Tailwind + Postgres. Tiene vistas de
    estimación, ingesta, Chunking Lab y ajustes. Es la capa que un cliente de negocio usaría.
- **Backend de negocio**
  - `cliente-web/src/` — servidor Express, rutas y clientes HTTP contra la API. Persiste sus propias
    estimaciones y runs en la base `estimator_web` (Postgres distinto del vectorial).
- **Servicio IA — FastAPI**
  - `app/main.py` — registra los routers con prefijo `/api/v1`; `GET /health` expone estado y
    proveedor LLM.
  - `app/routers/estimations.py` — camino **CAG** ya existente: `POST /api/v1/estimate`,
    `POST /api/v1/estimate/stream` y `GET /api/v1/context`.
  - `app/routers/sessions.py` — memoria conversacional: `POST /api/v1/sessions`,
    `GET /api/v1/sessions/{id}`, `POST /api/v1/sessions/{id}/estimate` y
    `POST /api/v1/sessions/{id}/estimate-acb`.
  - `app/routers/ingestion.py` — ingesta batch de fuentes del catálogo
    (`POST /api/v1/ingestion/runs`, `GET /api/v1/ingestion/jobs/{id}`).
  - `app/routers/search.py` — `POST /api/v1/search`.
  - `app/embedding_pipeline/router.py` — `POST /api/v1/embeddings/ingest` y
    `POST /api/v1/embeddings/compare`.

### Servicio IA, un nivel más abajo

- `app/ingestion/` — pipeline **offline** de datos: `catalog/`, `loaders/`, `parsers/` (JSON, TXT y
  transcripciones), `cleaning/`, `pii/`, `documents/`, `ports.py` y `orchestrator.py`. Transforma
  fuentes en `Document` y registra jobs; **no está conectado al retriever ni a la generación RAG**.
- `app/embedding_pipeline/structural.py` — `JSONStructuralChunker`: **un componente del presupuesto
  = un chunk**, con cabecera de contexto del padre (`project_summary`, sector, año, tecnología).
- `app/embedding_pipeline/embedder.py` — `OpenAIEmbedder` sobre `text-embedding-3-small` (1536
  dimensiones), batches de 100, reintentos y `embed_one` para queries.
- `app/embedding_pipeline/store/` — `DocumentRow` y `ChunkRow` contra `Base`; `ChunkStore` con
  búsqueda k-NN por distancia coseno (`<=>`). `metadata` es JSONB; el documento guarda
  `budget_id`, `client_sector` y `year`; cada chunk guarda `component_id`, `budget_id`,
  `client_sector`, `main_technology`, `year`, `complexity` y `estimated_hours` (del componente).
- `app/embedding_pipeline/ingest_service.py` — `RagIngestService`: chunk → embed → persist en una
  sola transacción; `409` si el `source_path` ya existe.
- `app/embedding_pipeline/retriever.py` — `SemanticRetriever`: embebe la query con el mismo modelo y
  pide los `k` chunks más cercanos. Sin filtros de metadata y sin reranking.
- `app/persistence/` — conviven el stack **sync** de la Sesión 06 (`psycopg`) y el **async** de la
  Sesión 08 (`asyncpg`). Alembic `0002_session8_pgvector` crea `documents` y `chunks`.
- `app/services/estimation.py` — `EstimationService` con el camino CAG: prompts Jinja2, wrapper LLM,
  validación Pydantic e `EstimationResult`. **Existe, pero no está conectado a los chunks
  recuperados.**
- `app/services/llm_wrapper.py` — wrapper LiteLLM con fallback, caché, coste y salida estructurada.
- `app/prompts/` — plantillas versionadas (`estimation/v1`, `metadata_extraction/v1`).

### Estado real del corpus y del índice

- `data/budgets_sample.json`: **17 presupuestos** y **60 componentes**; cuatro sectores
  (`finance`, `ecommerce`, `healthcare`, `industrial`); un chunk por componente.
- `text-embedding-3-small`, vectores de **1536 dimensiones**.
- En `main` conviven el baseline de la Sesión 08 y el material del directo (`scripts/rag_ops/`):
  la base de datos de desarrollo ya tiene índices **HNSW** (`vector_cosine_ops` y
  `halfvec_cosine_ops`) y `SEARCH_USE_HALFVEC=true` en `.env`. El diagnóstico debe reflejar lo que
  realmente usa la ruta de búsqueda; no afirmes “sin índice” si tu base ya lo tiene.
- El baseline canónico del trace es **17 documentos / 60 chunks**. Si aparece un documento de prueba
  manual (`manual-test::BUD-2024-001`), bórralo antes de trazar: contamina el corpus y no forma
  parte del seed.

### Lo que todavía no existe

- Una etapa que convierta la **transcripción cruda** en una consulta útil.
- Filtros por metadata en la recuperación (sector, tipo de proyecto, tecnología).
- Un paso que **reagrupe** los componentes por presupuesto y recupere los totales del padre.
- La conexión entre los chunks recuperados y `EstimationService` (augmentation + generation).
- Citas o trazabilidad de qué presupuesto fundamenta cada cifra de la estimación.

## Requisitos para el ejercicio

- Stack levantado desde `estimador-cag/`:

  ```bash
  docker compose up -d --build
  curl -s http://localhost:8000/health
  ```

  El servicio `api` aplica `alembic upgrade head` al arrancar. Si `/health` no responde, no
  empieces el trace.

- Corpus ingerido de forma idempotente:

  ```bash
  docker compose run --rm api python scripts/query_examples.py
  ```

  Los documentos ya presentes responden `409`; reejecutar no duplica.

- Clave de embeddings en `.env` (`OPEN_AI_KEY`) o un endpoint OpenAI-compatible
  (`EMBEDDING_BASE_URL` + `EMBEDDING_API_KEY`). El modelo debe ser `text-embedding-3-small`
  (`EMBEDDING_MODEL`), el mismo de la ingesta.

- `uv` disponible para ejecutar el script cliente con sus dependencias inline (`httpx`, `openai`).
  No hace falta añadir dependencias al proyecto.

- Docker y Docker Compose funcionando. Python 3.12 si ejecutas fuera del contenedor.

- No hace falta una base de datos viva para los tests: este ejercicio no toca `tests/`.

## Lo que entra y lo que no entra

**Entra:**

- Leer y describir la arquitectura real de las tres capas y del servicio IA.
- Ejecutar un trace manual y reproducible sobre `examples/transcripts/02_ambiguous.txt`.
- Embeber la transcripción completa y llamar a `POST /api/v1/search` con `k=5`.
- Pegar respuestas crudas y comentar la relevancia de cada chunk.
- Identificar cinco fallos concretos, anclados al trace.
- Proponer la evolución con un segundo diagrama de cajas y flechas.
- Crear `arquitectura-actual.md` en la raíz y subirlo en una rama/PR.

**No entra:**

- Reformulación de queries, expansión, HyDE, normalización o traducción implementada.
- Reranking, filtros de metadata implementados, búsqueda híbrida o un retriever nuevo.
- Módulo de generación, prompts de augmentation, citas o validación de la estimación.
- Endpoints nuevos o cambios en `/search`, `/embeddings/ingest` o `/embeddings/compare`.
- Cambios en `app/`, `frontend/`, `cliente-web/`, tests o migraciones.
- Tuning de HNSW, `ef_search`, `halfvec`, Postgres o embeddings.
- Métricas formales de retrieval (`recall@k`, NDCG) o evaluación con LLM juez.
- Copiar el diagrama RAG canónico de un blog o de un proveedor. La propuesta debe salir del trace.

## Datos para el ejercicio

Las transcripciones viven en `examples/transcripts/`:

| Fichero | Cliente / situación | Qué debe poner a prueba |
|---|---|---|
| `01_clear.txt` | Fintech que pide backend de banca móvil: OAuth 2.0/JWT, PSD2, SCA, Rails/Postgres/Redis. Alcance de fase uno cerrado. | Caso control: sector y tecnologías explícitos. Debería evocar `BUD-2024-001` (`finance`, `ruby_on_rails`). |
| `02_ambiguous.txt` | Ultramarinos gourmet que quiere vender online, fidelización, panel, pago con tarjeta y correo de confirmación. Divaga y no cierra alcance. | **Transcripción del trace.** Debería evocar `ecommerce`, sobre todo `BUD-2024-005` (storefront headless) y `BUD-2024-017` (MVP de ecommerce). |
| `03_hard.txt` | Clínica privada con agenda de citas, FHIR/HL7 y videoconsulta; el alcance cambia a mitad de reunión y no se cierra. | Caso difícil: sector `healthcare`, pero con tres alcances posibles y sin decisión. Debería evocar `BUD-2024-009` (citas + telemedicina + HL7/FHIR). |

`examples/transcripts/TEMPLATE.md` contiene la estructura del entregable y los headers de las cuatro
secciones obligatorias. `examples/trace_s09.py` es el cliente del trace: no importa `app.*`, no
modifica la base de datos y no añade comportamiento al servicio.

Corpus de referencia (`data/budgets_sample.json`):

- `BUD-2024-001` — fintech, `ruby_on_rails`, 480 h, 4 componentes.
- `BUD-2024-005` — ecommerce headless, `node`, 460 h, 4 componentes.
- `BUD-2024-009` — healthcare, `python_django`, 560 h, 4 componentes (agenda + telemedicina +
  HL7/FHIR).
- `BUD-2024-017` — ecommerce MVP, `node`, 160 h, 6 componentes de una línea.

Convención de `source_path` del seed: `data/budgets_sample.json::BUD-2024-XXX`. La metadata del
documento guarda `budget_id`, `client_sector` y `year`; la del chunk añade `component_id`,
`main_technology`, `complexity` y `estimated_hours` del componente. **No se persisten
`project_summary` ni `total_estimated_hours` del presupuesto padre**: tenlo en cuenta al razonar
sobre el rollup de la sección 3.

## ✍ Ejercicio

### Paso 0 — Preparar el corpus y verificar el estado

Antes de trazar, comprueba que el corpus es el canónico:

```bash
docker compose up -d --build
curl -s http://localhost:8000/health | jq

docker compose exec estimator-postgres psql -U estimator -d estimator \
  -c "SELECT count(*) AS documents FROM documents;" \
  -c "SELECT count(*) AS chunks FROM chunks;"
```

Debes ver **17 documentos / 60 chunks**. Si ves 18/64, revisa si queda el documento de prueba
`manual-test::BUD-2024-001`; bórralo (sus chunks caen por `ON DELETE CASCADE`) y vuelve a contar.
Después, ingesta de forma idempotente:

```bash
docker compose run --rm api python scripts/query_examples.py
```

La salida debe indicar que los 17 presupuestos ya están presentes (`409`) o que se han ingerido.
No sigas si el corpus no está completo.

### Paso 1 — Diagrama de la arquitectura actual

Dibuja la arquitectura de tres capas: **frontend**, **backend de negocio** y **servicio IA**. En el
servicio IA baja un nivel y muestra, como mínimo:

- `app/ingestion/` (catálogo, loaders, parsers, limpieza/PII, orquestador).
- `app/embedding_pipeline/` (`JSONStructuralChunker`, `OpenAIEmbedder`, `store/`, `RagIngestService`,
  `SemanticRetriever`).
- `app/persistence/` y las tablas `documents` / `chunks` con pgvector.
- Los endpoints HTTP que existen hoy: `/api/v1/embeddings/ingest`,
  `/api/v1/embeddings/compare`, `/api/v1/search` y el camino CAG de estimación
  (`/api/v1/estimate`, `/api/v1/sessions/{id}/estimate`).

Requisitos del diagrama:

- Marca con color, sombreado o anotación **dónde acaba lo implementado**.
- Muestra qué dato fluye entre módulos (presupuesto, chunks, vectores, query, `SearchResponse`).
- Señala con una flecha roja o un corte **en qué punto el flujo se queda corto** si llega una
  transcripción: la transcripción no entra en ningún endpoint de estimación y `/search` devuelve
  chunks, no una estimación.
- **No dibujes lo que falta**: eso es la sección 4. Si aparece una caja de generación en este
  diagrama, está mal.

Formato libre: Mermaid en un bloque de código, ASCII, boceto a mano o imagen exportada. Lo que
importa es que se entienda qué módulo habla con cuál y dónde muere el flujo.

### Paso 2 — Trace anotado de `02_ambiguous.txt`

El trace es manual y reproducible. Usa `examples/trace_s09.py` o comandos equivalentes; pega los
comandos exactos y la salida real en `arquitectura-actual.md`.

#### Paso 2.1 — Embeber la transcripción completa

```bash
export OPENAI_API_KEY="sk-..."   # o export OPEN_AI_KEY="sk-..."
uv run examples/trace_s09.py examples/transcripts/02_ambiguous.txt
```

El script hace dos cosas: (a) embebe el texto completo con el mismo modelo de la ingesta y (b)
llama a `POST /api/v1/search` con ese texto y `k=5`. No modifica el servicio.

Reporta en el entregable:

- modelo y dimensionalidad;
- norma L2;
- primera y última componente del vector;
- número de caracteres (y, si quieres, tokens aproximados);
- un comentario de una o dos frases sobre qué representa ese vector dado el contenido de la
  transcripción.

#### Paso 2.2 — Búsqueda semántica (`POST /api/v1/search`, k=5)

El script ya imprime el JSON crudo. Si prefieres `curl`, este es el equivalente:

```bash
curl -s -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  --data-binary "$(jq -Rs '{query: ., k: 5}' examples/transcripts/02_ambiguous.txt)" | jq
```

Pega la respuesta cruda con, al menos:

- `chunk_id`, `document_id`, `chunk_type`, `distance`;
- `content` (o un recorte legible);
- `metadata`: `budget_id`, `client_sector`, `component_id`, `main_technology`,
  `estimated_hours`.

#### Paso 2.3 — Lectura de los chunks devueltos

Para cada uno de los cinco chunks, comenta:

- a qué presupuesto histórico pertenece (`budget_id`);
- de qué sector es;
- si te parece relevante para lo que pide el cliente y por qué;
- qué falta en ese chunk para poder estimar (total del presupuesto, plazo, alcance, etc.).

Sé honesto: si el resultado es bueno, dilo; si no lo es, también. No basta con “no es relevante”:
cita la distancia y el componente.

##### Salida de referencia observada en este repo

Esta tabla **no es la respuesta del alumno**: sirve para detectar desviaciones groseras (modelo
distinto, corpus incompleto, endpoint equivocado). Las distancias pueden variar ligeramente con la
versión del modelo, el índice y el estado de la base.

```text
model           : text-embedding-3-small
characters      : 2813
dimensionality  : 1536
L2 norm         : 0.999817
first component : 0.006599
last component  : 0.019318
search_time_ms  : 556
results         : 5
```

| # | chunk_id | distance | budget_id | sector | componente | horas | ¿Qué se observa? |
|---|---------:|---------:|-----------|--------|------------|------:|------------------|
| 1 | 17 | 0.6110 | BUD-2024-005 | ecommerce | CART-002 · Cart and checkout service | 140 | Checkout y pago: encaja con “pagar con tarjeta”, pero es de un storefront headless mucho mayor que la tienda. |
| 2 | 16 | 0.6113 | BUD-2024-005 | ecommerce | CATALOG-001 · Product catalog API | 150 | Catálogo de producto: encaja; GraphQL + Elasticsearch es sobredimensionado para el cliente. |
| 3 | 18 | 0.6338 | BUD-2024-005 | ecommerce | RECO-003 · Personalized recommendations | 110 | Recomendaciones: no lo ha pedido; entra por similitud con el dominio ecommerce. |
| 4 | 19 | 0.6368 | BUD-2024-005 | ecommerce | STORE-004 · Storefront PWA | 60 | Storefront: encaja; PWA con SSR es más de lo que necesita. |
| 5 | 27 | 0.6430 | BUD-2024-008 | ecommerce | RET-001 · Returns portal | 140 | Devoluciones: no lo ha pedido; falso positivo por dominio ecommerce. |

Lectura de referencia: los cinco hits caen en una banda de **0.611–0.643** (Δ ≈ 0.032); cuatro de
cinco son del **mismo presupuesto** (`BUD-2024-005`, 460 h) y el quinto es otro presupuesto
ecommerce. No aparece `BUD-2024-017` (MVP de 160 h), que es probablemente el que más se parece a la
escala real del cliente. Ningún hit trae el total de horas del presupuesto padre ni el resumen del
proyecto: solo el componente y sus horas.

### Paso 3 — Diagnóstico: cinco fallos identificados

Enumera **cinco fallos** que impidan hoy convertir la transcripción en una estimación de calidad.
Para cada uno redacta exactamente tres líneas:

- **Problema observado** — qué ves que pasa, citando el trace (chunk, distancia, sector, presupuesto).
- **Causa probable** — qué decisión arquitectónica o pieza ausente lo provoca.
- **Propuesta de solución** — qué pieza, etapa o cambio lo resolvería, sin implementarlo.

Los fallos deben ser concretos y verificables. Ejemplos de lo que **no** vale:

- “El sistema no es bueno.”
- “Falta RAG.”
- “Los embeddings no entienden español.”

Ejemplos de lo que **sí** vale:

- “Los cinco chunks caen entre 0.611 y 0.643: la consulta es la media de cinco intenciones y el
  sistema no tiene una opinión fuerte.”
- “Cuatro de los cinco chunks son de `BUD-2024-005`, un presupuesto de 460 h, cuando el cliente pide
  algo mucho más pequeño; no hay filtro por escala ni por tipo de proyecto.”
- “El hit 1 es `CART-002` (140 h) y no incluye el total de 460 h del presupuesto padre; con ese
  chunk no se puede fundamentar un coste.”

Señales que el trace de este repo permite defender (no las copies sin comprobarlas en tu salida):

1. **Transcripción cruda como query.** El vector es la media de la tienda física, la fidelización,
   el panel, el pago, el correo y la anécdota de Francia. No representa “qué quiere construir”.
2. **Desajuste de idioma y registro.** La transcripción es español conversacional; el corpus es
   inglés técnico. El match llega por términos sueltos (“vender por internet” → storefront,
   “pagar con tarjeta” → cart/checkout), no por intención.
3. **Recuperación sin filtro de metadata.** El top-5 se concentra en un presupuesto sobredimensionado
   y no usa `client_sector`, `main_technology` ni el tipo de proyecto inferido.
4. **No hay augmentation + generation.** La última salida viva es una lista de chunks con
   distancias. `EstimationService` existe, pero no recibe esos chunks ni produce una estimación
   fundamentada.
5. **Granularidad de componente sin rollup.** El chunk es un componente suelto; no hay `budget_id`
   con `total_estimated_hours`, `project_summary` ni plazo. El dato que hace falta para estimar no
   está en el contexto recuperado.

Si encuentras más de cinco, elige los cinco más relevantes y deja el resto en una sección
**Otros** al final. Candidatos menores: `k=5` fijo sin umbral de distancia, ingesta y retrieval
desconectados del camino CAG, idempotencia solo por `source_path`, o HNSW/halfvec irrelevante a
esta escala.

### Paso 4 — Propuesta de evolución arquitectónica

Dibuja un segundo diagrama con la misma arquitectura de tres capas, pero ahora con las cajas y
módulos que añadirías al servicio IA para cerrar el flujo **transcripción → estimación generada**.
No es una propuesta de implementación: son cajas, flechas y nombres.

Requisitos:

- Marca claramente qué cajas son **nuevas** respecto a la sección 1.
- Conserva y reutiliza lo que ya existe: `OpenAIEmbedder`, `SemanticRetriever`/`ChunkStore`,
  `documents`/`chunks`, `EstimationService`, `LLMWrapper`, `EstimationResult`.
- Muestra el dato que fluye entre las cajas: transcripción → brief/consulta normalizada → filtros →
  chunks → contexto ensamblado → `EstimationResult`.
- La propuesta debe salir de los fallos del trace, no de un diagrama genérico. Si tu sección 4 es
  solo “Query → Retrieval → Augmentation → Generation” sin decir qué entra y qué sale de cada caja,
  no has hecho el ejercicio.

Cajas que el diagnóstico debería justificar (nómbralas como quieras, pero que existan y tengan una
responsabilidad clara):

- **Comprensión / normalización de la transcripción** — destila la reunión en un brief estructurado
  (tipo de proyecto, sector, features, restricciones) y una consulta en el registro del corpus.
- **Retrieval con filtros** — extiende la búsqueda actual con metadata (`client_sector`,
  `main_technology`, tipo de proyecto) y, si hace falta, con criterios de escala.
- **Ensamblado de contexto / augmentation** — reagrupa los chunks por `budget_id`, recupera o
  persiste los totales del presupuesto padre y construye un contexto acotado y citable.
- **Generación fundamentada** — reutiliza `EstimationService`/`LLMWrapper` para producir un
  `EstimationResult` validado, con referencia a los presupuestos que lo fundamentan.

Cierra con un párrafo de **máximo diez líneas** que responda:

1. ¿Cuál es la responsabilidad de cada módulo nuevo?
2. ¿Qué dato fluye entre ellos?
3. ¿Qué pieza es la más crítica — la que atacarías primero si solo pudieras construir una?

### Entregable

Un único archivo **`arquitectura-actual.md` en la raíz de `estimador-cag/`**, con las cuatro
secciones anteriores. Debe incluir:

- los dos diagramas, claramente distinguibles;
- comandos ejecutables y respuestas **reales** del sistema en la sección 2;
- los cinco fallos con las tres líneas pedidas;
- la propuesta de evolución y el párrafo final.

Súbelo en una rama nueva **`session-09/pre-work`** y abre PR contra `main`. No hace falta mergearlo
antes de la sesión en vivo; el branch basta.

## Criterios de "hecho"

El ejercicio está completo cuando:

- [ ] Existe `arquitectura-actual.md` en la raíz con las cuatro secciones obligatorias.
- [ ] La sección 1 muestra las tres capas y, en el servicio IA, `app/ingestion/`,
      `app/embedding_pipeline/`, `app/persistence/` y los endpoints actuales.
- [ ] La sección 1 marca dónde acaba lo implementado y no dibuja el módulo de generación.
- [ ] La sección 2 incluye los comandos exactos y la respuesta real del sistema.
- [ ] El paso 1 del trace reporta dimensionalidad (1536), norma, primera y última componente.
- [ ] El paso 2 pega los cinco chunks crudos con distancias y metadata.
- [ ] El paso 3 comenta cada chunk: presupuesto, sector, relevancia y qué falta para estimar.
- [ ] La sección 3 contiene exactamente cinco fallos, cada uno con Problema / Causa / Propuesta.
- [ ] Cada fallo cita una observación del trace; no hay afirmaciones genéricas.
- [ ] La sección 4 tiene un diagrama distinto de la sección 1, con cajas nuevas marcadas.
- [ ] El párrafo final tiene diez líneas o menos y responde a las tres preguntas.
- [ ] No hay cambios en `app/`, endpoints, tests, migraciones ni en el retriever.
- [ ] El PR contiene solo el documento y, si se añaden, las transcripciones/script de ejemplo.

## Checklist de verificación

- [ ] `examples/transcripts/01_clear.txt`, `02_ambiguous.txt` y `03_hard.txt` existen y son texto
      legible en español.
- [ ] `examples/transcripts/TEMPLATE.md` conserva los cuatro headers del entregable.
- [ ] `examples/trace_s09.py` no importa `app.*` ni escribe en Postgres.
- [ ] El script llama a `/api/v1/search` (no a `/search` sin prefijo) y usa `k=5`.
- [ ] El script usa `text-embedding-3-small` o el `EMBEDDING_MODEL` configurado; no mezcla modelos.
- [ ] El corpus del trace es 17 documentos / 60 chunks; no hay `manual-test::` en `documents`.
- [ ] La salida del trace es real: no hay distancias inventadas ni placeholders sin rellenar.
- [ ] Los diagramas usan Mermaid/ASCII/imagen y se renderizan sin errores.
- [ ] La sección 4 no reutiliza literalmente el diagrama de la sección 1: añade cajas nuevas y
      flechas de datos.
- [ ] La propuesta menciona al menos una pieza existente que se reutiliza (`Embedder`, `ChunkStore`,
      `EstimationService`, `LLMWrapper` o `EstimationResult`).
- [ ] `git diff --stat` no muestra cambios en `app/`.
- [ ] La rama se llama `session-09/pre-work` y el PR apunta a `main`.

## Documentación de referencia

- Persistencia y búsqueda de la Sesión 08:
  [`docs/persistencia-pgvector-busqueda-semantica-sesion-08.md`](../docs/persistencia-pgvector-busqueda-semantica-sesion-08.md)
- Operación de índices y `halfvec` de la Sesión 08 en vivo:
  [`docs/sesion-8-live-indexacion-vectorial.md`](../docs/sesion-8-live-indexacion-vectorial.md)
- Spec de persistencia y búsqueda:
  [`specs/sesion-8-persistencia-pgvector-busqueda-semantica.md`](./sesion-8-persistencia-pgvector-busqueda-semantica.md)
- Pipeline de embeddings de la Sesión 07:
  [`specs/sesion-7-pipeline-embeddings.md`](./sesion-7-pipeline-embeddings.md)
- pgvector — operadores de distancia y operator classes:
  https://github.com/pgvector/pgvector
- OpenAI — embeddings (`text-embedding-3-small`):
  https://platform.openai.com/docs/guides/embeddings
- FastAPI — `Depends`, routers y `response_model`:
  https://fastapi.tiangolo.com/
- SQLAlchemy 2.0 — `AsyncSession` y `async_sessionmaker`:
  https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html

## Nota

**Por qué este pre-work no implementa nada.** La tentación al ver el trace es arreglarlo: meter un
filtro, reformular la query, llamar al LLM. Eso es exactamente lo que la Sesión 09 hará en vivo. El
valor de este ejercicio está en **ver el sistema tal como es** y en formular preguntas concretas
desde la evidencia. Un diagnóstico honesto vale más que un parche rápido.

**Qué es nuestro (y por qué).** Este repo no es el diagrama genérico de un proveedor. El servicio IA
tiene `app/ingestion/` para el pipeline batch, `app/embedding_pipeline/` para chunking, embeddings,
store y retrieval, y `app/persistence/` con dos stacks (sync de la Sesión 06 y async de la Sesión
08). La estimación CAG ya existe (`EstimationService`, `LLMWrapper`, `EstimationResult`) pero vive
en un camino separado del retriever. El corpus es el seed real de 17 presupuestos / 60 componentes,
con cuatro sectores y chunks de un componente. La base de desarrollo incluye HNSW y `halfvec` del
directo de la Sesión 08. El diagnóstico debe describir **esto**, no un `storage/` genérico ni un
`generation/rag/` importado de otro proyecto.

**Sobre la salida de referencia.** La tabla de la sección 2.3 se obtuvo ejecutando el trace contra el
corpus de este repo. Las distancias pueden moverse unos milésimos si cambias el modelo, el índice o
el estado de la base. Úsala para comprobar que estás mirando el sistema correcto, no para copiarla:
tu entregable debe tener **tu** salida real y **tus** comentarios.

**Qué NO hay que hacer.** No implementes reformulación de queries, ni reranking, ni un retriever
nuevo, ni filtros de metadata, ni el módulo de generación. No modifiques el endpoint de búsqueda
actual. No crees endpoints. No busques en internet la arquitectura RAG canónica para copiarla en la
sección 4: la propuesta tiene que salir del trace y de los fallos. Si la sección 4 acaba siendo
“Query → Retrieval → Augmentation → Generation” sin más, no estás haciendo el ejercicio: estás
repitiendo terminología.

**Convención de entrega.** Rama `session-09/pre-work`, PR contra `main`, un único
`arquitectura-actual.md` en la raíz. El PR puede quedar sin merge antes de la sesión en vivo; el
branch basta.
