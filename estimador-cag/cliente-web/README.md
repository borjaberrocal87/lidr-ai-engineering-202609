# cliente-web

Cliente de negocio del estimador: una app **Node/Express + EJS** que ofrece el formulario, persiste las estimaciones en **Postgres** y consume la API FastAPI (`estimador-cag`) por HTTP. Nunca habla directamente con OpenAI/Anthropic: las claves viven solo en el backend Python.

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
│   ├── app.js               # createApp({ repo, estimator }) (inyección para tests)
│   ├── config.js            # Variables de entorno
│   ├── constants.js         # Enums y umbrales compartidos con la API
│   ├── estimatorClient.js   # Cliente HTTP + errores tipados (400/422/502/503)
│   ├── db.js                # Repositorio Postgres + repositorio en memoria
│   ├── migrate.js           # Aplica migrations/*.sql
│   └── routes/estimations.js
├── views/                   # EJS (formulario, listado, detalle)
├── public/                  # CSS compilado, JS del formulario y assets LIDR
├── migrations/              # SQL versionado
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
- `GET /health` — estado del servicio.

## Variables de entorno

| Variable | Descripción | Default |
|---|---|---|
| `PORT` | Puerto del cliente | `3000` |
| `ESTIMATOR_API_BASE_URL` | URL base de la API FastAPI | `http://localhost:8000` |
| `ESTIMATOR_TIMEOUT_MS` | Timeout de la llamada a la API | `60000` |
| `DATABASE_URL` | PostgreSQL (vacío = repositorio en memoria) | — |
| `NODE_ENV` | Entorno | `development` |

## Tests

```bash
npm test
```

Los tests usan `node:test` + `supertest` con un repositorio en memoria y un estimador falso: no requieren red ni base de datos.

## Docker

El servicio `cliente-web` y `postgres` están definidos en el `docker-compose.yml` del proyecto padre. Desde `estimador-cag/`:

```bash
docker compose up --build -d
# cliente-web: http://localhost:3000
```
