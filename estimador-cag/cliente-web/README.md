# cliente-web

Cliente de negocio del estimador: una app **Node/Express + EJS** que ofrece el formulario, persiste las estimaciones en **Postgres** y consume la API FastAPI (`estimador-cag`) por HTTP. Nunca habla directamente con OpenAI/Anthropic: las claves viven solo en el backend Python.

Desde la sesión 6 incluye además una pantalla de **Ingesta de datos** que consume los endpoints `/api/v1/ingestion/*` del backend (catálogo de fuentes, lanzamiento de ingestas y estado de los jobs). El cliente no persiste nada de la ingesta: la API Python es la dueña de los jobs.

Desde la sesión 7 incorpora **`/chunking-lab`** (comparativa de las ocho estrategias de chunking, con historial persistido en `chunking_comparisons`) y **`/ai-settings`** (overrides de modelo en runtime vía `GET/PUT /api/v1/config/models`). La navbar muestra el modelo primario efectivo como badge, con caché de 15 s y degradación silenciosa si la API no responde.

El look & feel replica el tema de LIDR Academy (oscuro, tipografía Helvetica, amarillo de marca `#F6DE82`, sin sombras) definido como tema de Tailwind.

## Requisitos

- Node 20+
- La API `estimador-cag` corriendo (por defecto en `http://localhost:8000`)
- Postgres (opcional: sin `DATABASE_URL` usa un repositorio en memoria, sin persistencia)

## Puesta en marcha

```bash
npm install
npm run build:css        # genera public/styles.css (ya viene generado en el repo)
cp .env.example .env
npm start                # http://localhost:3000
```

Con Postgres, aplica la migración antes (o deja que `npm start` la aplique sola):

```bash
DATABASE_URL=postgres://estimator:estimator@localhost:5432/estimator_web npm run migrate
```

## Estructura

```
cliente-web/
├── src/
│   ├── server.js            # Arranque: migra Postgres y levanta Express
│   ├── app.js               # createApp({ repo, estimator, ingestion }) (inyección para tests)
│   ├── config.js            # Variables de entorno
│   ├── constants.js         # Enums y umbrales compartidos con la API
│   ├── estimatorClient.js   # Cliente HTTP + errores tipados (400/422/502/503)
│   ├── ingestionClient.js   # Cliente HTTP de /ingestion (catálogo, runs, jobs)
│   ├── embeddingsClient.js  # Cliente de /embeddings/compare (timeout 600 s)
│   ├── configClient.js      # Cliente de /config/models + caché del badge
│   ├── chunkingConstants.js # Catálogo de estrategias para la UI
│   ├── aiSettings.js        # Knobs y etiquetas de Ajustes
│   ├── corpus.js            # Carga del corpus del Chunking Lab
│   ├── db.js                # Repositorio Postgres + repositorio en memoria
│   ├── chunkingRepo.js      # Runs del Chunking Lab (Postgres / memoria)
│   ├── migrate.js           # Aplica migrations/*.sql en orden
│   └── routes/              # estimations.js, ingestion.js, chunkingLab.js, aiSettings.js
├── views/                   # EJS (estimación, ingesta, chunking-lab, ai-settings)
├── data/                    # Copia del corpus (sincronizada con ../data/)
├── public/                  # CSS compilado, JS del formulario y assets LIDR
├── migrations/              # SQL versionado (estimaciones + chunking_comparisons)
├── test/                    # node:test + supertest
├── Dockerfile
├── tailwind.config.js
└── package.json
```

## Rutas

- `GET /` — formulario (subida de `.txt`, selectores de tipo/detalle/formato y versión de prompt).
- `POST /estimations` — llama a la API, persiste y redirige al detalle. Un `400` de guardrail o un `422` se re-renderizan en el formulario con el motivo.
- `GET /estimations` — histórico.
- `GET /estimations/:id` — detalle (summary, tarjetas de duración/coste/confianza y tabla de fases).
- `GET /ingestion` — catálogo de fuentes (con su decisión y si contienen PII) y últimos jobs; botón “Ingerir” en las fuentes `include`.
- `POST /ingestion/runs` — lanza una ingesta (`source_name`) y redirige al detalle del job.
- `GET /ingestion/jobs/:id` — estado del job (se refresca solo mientras está `pending`/`running`).
- `GET /chunking-lab` — historial de comparativas.
- `GET /chunking-lab/new` — formulario de estrategias, queries y top-k.
- `POST /chunking-lab` — ejecuta la comparativa y persiste el run.
- `GET /chunking-lab/:id` — estadísticos, coste y top-k por query.
- `GET /ai-settings` — ajustes de modelos (effective/default/overridden).
- `POST /ai-settings` — guarda overrides (`""` = volver al default de `.env`).
- `GET /health` — estado del servicio.

## Variables de entorno

| Variable | Descripción | Default |
|---|---|---|
| `PORT` | Puerto del cliente | `3000` |
| `ESTIMATOR_API_BASE_URL` | URL base de la API FastAPI | `http://localhost:8000` |
| `ESTIMATOR_TIMEOUT_MS` | Timeout de la llamada a la API | `60000` |
| `CHUNKING_TIMEOUT_MS` | Timeout de `/embeddings/compare` | `600000` |
| `CONFIG_TIMEOUT_MS` | Timeout del badge/Ajustes | `3000` |
| `CONFIG_CACHE_TTL_MS` | TTL de la caché del badge | `15000` |
| `DATABASE_URL` | PostgreSQL (vacío = repositorio en memoria) | — |
| `NODE_ENV` | Entorno | `development` |

## Tests

```bash
npm test
```

Los tests usan `node:test` + `supertest` con repositorios en memoria y clientes falsos: no requieren red ni base de datos. Cubren también la sincronización del corpus, los clientes de embeddings/config y las rutas de Chunking Lab y Ajustes.

Documentación completa de ambas funcionalidades:
[`../docs/chunking-lab-y-ajustes-runtime-sesion-07.md`](../docs/chunking-lab-y-ajustes-runtime-sesion-07.md).

## Docker

El servicio `cliente-web` y `postgres` están definidos en el `docker-compose.yml` del proyecto padre. Desde `estimador-cag/`:

```bash
docker compose up --build -d
# cliente-web: http://localhost:3000
```
