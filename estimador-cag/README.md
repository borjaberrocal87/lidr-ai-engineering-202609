# estimador-cag

API FastAPI que recibe una **descripción de proyecto tipada** (descripción libre + tipo, nivel de detalle y formato de salida) y devuelve una estimación de software generada por un LLM, usando **arquitectura CAG** (Context-Augmented Generation): el contexto estático (ejemplos de estimaciones previas) se inyecta directamente en el prompt en cada llamada. Sin base de datos, sin retrieval, sin persistencia.

El prompt no vive en el código: se compone con **plantillas Jinja2 versionadas** (`app/prompts/`), de modo que cambiar texto, ejemplos o reglas es cambiar un `.j2`, no refactorizar el servicio.

## Alcance

El contexto CAG son los ejemplos few-shot de `app/prompts/estimation/v1/examples.j2`, inyectados en el system prompt en cada llamada. Sin base de datos ni retrieval: es una decisión, no deuda pendiente.

El umbral para revisarla: cuando el catálogo crezca más allá de unos 15-20 ejemplos, toca migrar a una fuente vectorial (RAG). El punto de cambio está aislado en el template (`examples.j2`) y en el loader, de modo que el router y el servicio no se enteran.

## Estructura

```
estimador-cag/
├── app/                         # Backend (FastAPI): aquí vive el LLM y las claves
│   ├── main.py                  # App FastAPI, router /api/v1, GET /health, /docs
│   ├── config.py                # Settings (Pydantic BaseSettings) desde .env
│   ├── logging_config.py        # Configuración de structlog (console/json)
│   ├── routers/
│   │   └── estimations.py       # /estimate, /estimate/stream (SSE) y /context
│   ├── schemas/
│   │   └── estimations.py       # Contrato HTTP (Pydantic) y enums del formulario
│   ├── prompts/                 # Prompts versionados (Jinja2)
│   │   ├── loader.py            # render_estimation_prompt(request, version)
│   │   └── estimation/v1/       # system.j2, user.j2, examples.j2
│   ├── services/
│   │   ├── llm_service.py       # Orquesta: valida, renderiza prompts y llama al wrapper
│   │   └── llm_wrapper.py       # Wrapper LiteLLM (fallback, caché, coste, logging)
├── frontend/                    # Capa de presentación (no importa `app.*`)
│   ├── config.py                # API_BASE_URL y timeouts
│   ├── client.py                # Cliente HTTP (httpx) contra la API
│   ├── models.py                # Modelos locales de respuesta
│   ├── logging_config.py        # Logging del frontend
│   └── streamlit_app.py         # Formulario tipado (Streamlit)
├── tests/                       # Tests con pytest (proveedores y API mockeados)
│   └── prompts/                 # Tests de los templates (sin LLM)
├── examples/
│   └── transcripcion.md         # Transcripción de reunión de ejemplo (input del formulario)
├── specs/
│   ├── sesion-2-scaffolding-fastapi.md  # Spec del backend FastAPI (sesión 2)
│   ├── sesion-3-interfaz-conversacional-streamlit.md  # Spec de la UI (sesión 3)
│   ├── sesion-4-formulario-tipado-prompt-jinja2.md  # Spec del formulario y prompts (sesión 4)
│   └── proveedor-openai-compatible.md  # Spec del proveedor custom (OpenAI-compatible)
├── Dockerfile               # Build multi-stage (builder / test / runtime)
├── docker-compose.yml       # Servicios api, ui y test
├── .dockerignore
├── .env.example
├── pyproject.toml
└── README.md
```

> Arquitectura por capas: el `frontend/` solo habla con el backend por HTTP (`API_BASE_URL`) y nunca importa `app.*`. Cambiar Streamlit por otra UI consiste en reescribir el punto de entrada reutilizando `frontend.client`. Un test (`tests/test_frontend_decoupling.py`) lo verifica.

> El pipeline de CI vive en la raíz del repo: `.github/workflows/ci.yml` (lint, type-check, tests y smoke test de Docker).

## Requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) como gestor de paquetes
- API key de [OpenAI](https://platform.openai.com/) y/o [Anthropic](https://console.anthropic.com/)

## Puesta en marcha

```bash
# 1. Instalar dependencias
uv sync

# 2. Configurar variables de entorno
cp .env.example .env
# Edita .env y añade tu API key y el proveedor deseado (LLM_PROVIDER)

# 3. Arrancar el servidor
uv run uvicorn app.main:app --reload
```

Documentación Swagger: http://localhost:8000/docs

## Uso

El body es un `EstimationRequest` tipado: una descripción libre y tres knobs cerrados por `Enum` (`project_type`, `detail_level`, `output_format`).

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "En la reunión con el equipo de marketing, el cliente explicó que necesita una landing page con formulario de contacto, integración con su CRM actual (HubSpot), y una sección de blog con editor WYSIWYG. El plazo ideal sería tenerlo listo en 4 semanas.",
    "project_type": "web_saas",
    "detail_level": "detailed",
    "output_format": "phases_table"
  }'
```

Respuesta:

```json
{
  "estimation": "## Estimación: ...",
  "prompt_version": "v1",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "temperature": 0.2,
  "input_tokens": 1234,
  "output_tokens": 567,
  "truncated": false
}
```

Los valores válidos de los enums son:

- `project_type`: `mobile_app`, `web_saas`, `internal_tool`, `data_pipeline`.
- `detail_level`: `summary`, `medium`, `detailed`.
- `output_format`: `phases_table`, `line_items`, `narrative`.

Pydantic los valida en el borde: un valor desconocido devuelve **422**. `prompt_version` indica el template que produjo la estimación.

`truncated: true` significa que el modelo agotó `LLM_MAX_TOKENS` antes de terminar y la estimación puede estar incompleta. En ese caso la API responde igualmente `200` (para no perder la respuesta parcial), pero lo indica de forma explícita y deja un warning en los logs.

Errores de `POST /api/v1/estimate`:

- `503` — falta la configuración del proveedor activo (p. ej. la API key). El mensaje nombra la variable de entorno.
- `502` — el proveedor LLM falló (red, timeout, rate limit o estado). El detalle interno se registra en el servidor y **no** viaja al cliente.
- `500` — error inesperado en el código.

Estado del servicio:

```bash
curl http://localhost:8000/health
```

`/health` responde siempre `200` (no depende del LLM) e incluye `llm_configured` para saber si el proveedor activo tiene credenciales.

### Estimación en streaming (SSE)

Además del endpoint bloqueante, `POST /api/v1/estimate/stream` expone la generación token a token como **Server-Sent Events**. El formulario del frontend usa el endpoint no-streaming, pero cualquier cliente HTTP puede consumir el stream con el mismo body tipado:

```bash
curl -N -X POST http://localhost:8000/api/v1/estimate/stream \
  -H "Content-Type: application/json" \
  -d '{"description": "En la reunión...", "project_type": "web_saas", "detail_level": "medium", "output_format": "phases_table"}'
```

```text
event: token
data: {"text": "## Estimación"}

event: token
data: {"text": "..."}

event: done
data: {"prompt_version":"v1","model":"gpt-4o-mini","provider":"openai","input_tokens":1234,"output_tokens":567,"truncated":false}
```

- `token` — delta de texto.
- `done` — métricas de la llamada (versión de prompt, modelo, proveedor, tokens, truncado).
- `error` — fallo de configuración del proveedor o de generación (el `200` ya se envió, así que no se puede cambiar el estado).

La descripción se valida con **422** (Pydantic) antes de abrir el flujo. Un proveedor mal configurado no devuelve `503` en este endpoint: se emite como evento `error` (la UI ya avisa antes con `llm_configured` de `GET /api/v1/context`). El endpoint usa el SSE nativo de FastAPI (`fastapi.sse`), que añade pings de keepalive automáticos.

### Contexto CAG

`GET /api/v1/context` devuelve el system prompt **renderizado** con la versión por defecto y los límites de la descripción, para que la UI muestre la información sin importar código del backend:

```bash
curl http://localhost:8000/api/v1/context
```

## Límites de la llamada al LLM

La llamada al proveedor está acotada por configuración:

- `LLM_TIMEOUT_SECONDS` (por defecto `30`) y `LLM_MAX_RETRIES` (por defecto `2`) se aplican al cliente del SDK, para no heredar el timeout por defecto de diez minutos.
- `LLM_MAX_TOKENS` (por defecto `2048`) acota el coste de salida y se traduce en `truncated` cuando la respuesta se corta.
- `DESCRIPTION_MIN_LENGTH` / `DESCRIPTION_MAX_LENGTH` (por defecto `20` / `50000` caracteres) se validan en el borde (422) antes de gastar un token. El servicio repite la comprobación por si lo invoca otro adaptador (worker, CLI); en ese caso responde `422`.

El `user.j2` envuelve la descripción en el bloque `<project_description>` y el `system.j2` explica al modelo que ese bloque son datos, no instrucciones, con la sección `<limite_de_datos>`.

## Formulario (Streamlit)

Además de la API, el proyecto incluye un formulario web que construye un `EstimationRequest` tipado y lo envía a `POST /api/v1/estimate`, sin usar `curl`, Postman ni Swagger. Es una **capa de presentación independiente**: consume la API por HTTP y no importa código del backend, así que puede sustituirse por otra UI reutilizando `frontend/client.py`.

Arranca primero la API y luego la UI (en otra terminal):

```bash
# 1) Backend
uv run uvicorn app.main:app --reload

# 2) Frontend
uv run streamlit run frontend/streamlit_app.py
```

Se abre en http://localhost:8501. El frontend apunta al backend por `API_BASE_URL` (por defecto `http://localhost:8000`). El formulario (`st.form`) ofrece un textarea para la descripción y selectores para tipo de proyecto, nivel de detalle y formato de salida; los selectores envían los strings de los enums. El panel lateral muestra el system prompt renderizado y las métricas de la última llamada (versión de prompt, modelo, proveedor, tokens, coste, caché y fallback), obtenidas de `GET /api/v1/context` y de la respuesta de `/estimate`.

Las claves LLM ya no las lee la UI: viven solo en el backend. Si el proveedor no está configurado, la UI lo avisa (`llm_configured`).

## Logging

Las llamadas al LLM se registran con [structlog](https://www.structlog.org/) (integrado con `logging`) en la **API**, que es quien habla con el proveedor. El frontend solo registra su propia actividad con `logging` estándar. Cada llamada al LLM emite:

- `llm.call.start` / `llm.stream.start` — modelo, proveedor y `max_tokens`.
- `llm.call.end` / `llm.stream.end` — tokens de entrada/salida, coste estimado (`cost_usd`), si se usó el fallback (`fallback_used`) y latencia en milisegundos.
- `cache.hit` / `cache.miss` / `cache.stored` — resultado de la caché de respuestas.
- `llm.call.error` / `llm.stream.error` — error con traceback y latencia.
- `llm.response.truncated` (en el router) — warning cuando el modelo agotó `LLM_MAX_TOKENS`.

```text
2026-09-17T15:04:54.945815Z [info] llm.call.start model=gpt-4o-mini provider=openai max_tokens=2048
2026-09-17T15:05:03.123456Z [info] llm.call.end   model=gpt-4o-mini provider=openai input_tokens=1234 output_tokens=567 cost_usd=0.000525 fallback_used=False latency_ms=8421.3
```

Cada petición recibe un `request_id` (de la cabecera `X-Request-ID` o generado) que se propaga por `structlog.contextvars` a **todos** los logs de esa petición y se devuelve en la respuesta, de modo que se puede reconstruir un flujo completo de extremo a extremo.

Por privacidad se registran **solo metadatos**: nunca la API key ni el texto de la descripción (posible información confidencial del cliente). El nivel se controla con `LOG_LEVEL` y el formato con `LOG_FORMAT` (`console` para texto legible, `json` para agregadores). En Docker los logs salen por stdout y se consultan con `docker compose logs -f api` (o `ui`).

## Wrapper de proveedores y caché

La llamada al proveedor vive en `app/services/llm_wrapper.py`, un wrapper sobre **LiteLLM** que unifica los tres proveedores (`openai`, `anthropic` y `custom` OpenAI-compatible) y añade:

- **Fallback**: el `Router` gestiona primario y secundario según `LLM_ROUTING_MODE`.
  - `fallback` (por defecto): el primario (`LLM_MODEL`) se usa siempre y solo se cae a `LLM_FALLBACK_MODEL` si el primario lanza una excepción tras los reintentos.
  - `balanced`: ambas deployments comparten `model_name`, así que LiteLLM reparte las peticiones entre primario y secundario (`simple-shuffle`); útil para comparar modelos.
  En ambos casos la respuesta indica `fallback_used` (en `balanced`, significa "atendido por el deployment secundario").
- **Caché exact-match** (`app/services/cache.py`): la clave es un SHA-256 de un *material de clave* canónico —el `EstimationRequest`, la `prompt_version` y la huella (`prompt_fingerprint`) de las fuentes de esa versión— más los knobs de generación (`model`, `max_tokens`, `temperature`). El adaptador de caché no conoce la semántica del prompt: el dominio le pasa material opaco. Cualquier cambio relevante (campos del request, versión, o edición de un `.j2`) invalida la caché solo.
  - `CACHE_BACKEND=memory` (por defecto) usa una caché TTL en proceso, sin infraestructura.
  - `CACHE_BACKEND=redis` usa Redis (persistente y compartida) con `REDIS_URL`/`CACHE_TTL`.
  - `CACHE_BACKEND=none` la desactiva.
- **Coste estimado** (`cost_usd`) a partir de una tabla de precios por millón de tokens.

`POST /api/v1/estimate` y el evento SSE `done` exponen `cache_hit`, `cost_usd` y `fallback_used`.

Para ver la caché en acción, lanza dos veces la misma petición:

```bash
curl -s localhost:8000/api/v1/estimate -H 'Content-Type: application/json' \
  -d '{"description": "Necesitamos un CRM con auth, contactos y roles. MVP en seis semanas.", "project_type": "web_saas", "detail_level": "medium", "output_format": "phases_table"}' \
  | jq '{cache_hit, cost_usd, prompt_version}'
```

Con Redis (`CACHE_BACKEND=redis`) puedes inspeccionar las claves:

```bash
docker compose exec redis redis-cli KEYS 'estimation:*'
```

## Prompts versionados

El prompt ya no es un `f-string` en el código. Vive en plantillas Jinja2 con versiones en disco:

```
app/prompts/
├── loader.py                 # render_estimation_prompt(request, version="v1") -> (system, user)
└── estimation/
    └── v1/
        ├── system.j2         # rol, reglas, condicionales de formato y detalle
        ├── user.j2           # envuelve la descripción en <project_description>
        └── examples.j2       # ejemplos few-shot, incluidos con {% include %}
```

El `Environment` de Jinja2 usa `FileSystemLoader` sobre `app/prompts/`, `StrictUndefined` (un typo entre el contexto y la plantilla revienta en el render, no se interpola vacío) y `trim_blocks`/`lstrip_blocks` (las etiquetas de control no dejan saltos ni espacios en el prompt). `system.j2` decide el bloque de `output_format` y el de `detail_level` con `{% if %}` e incluye los ejemplos con `{% include "estimation/v1/examples.j2" %}`.

`render_estimation_prompt` devuelve `(system, user)` por separado, que es lo que el wrapper envía como dos mensajes (`role: "system"` y `role: "user"`). La respuesta incluye `prompt_version`.

Para añadir una versión, duplica `v1/` como `v2/`, edita las plantillas y llama a `render_estimation_prompt(request, version="v2")`. La convención `v1/`, `v2/` no es opcional: permite comparar, ensayar y volver atrás, y el `prompt_version` de la respuesta dice qué prompt produjo cada estimación. La clave de caché incluye la versión y la huella de las fuentes del prompt, así que cambiar de versión (o editar un `.j2`) invalida la caché sola.

## Transcripción de ejemplo

En `examples/transcripcion.md` hay una transcripción de reunión realista (landing page + integración HubSpot + blog con editor WYSIWYG) lista para usar como `description` del formulario (o del body de `POST /api/v1/estimate`).

## Tests

```bash
uv run pytest
```

Los tests mockean los proveedores LLM (no hacen llamadas reales) y cubren:

- el endpoint `/api/v1/estimate` y los schemas de entrada/salida (enums tipados, validación de longitud y que una entrada inválida **no** llega a invocar al LLM) (`tests/test_estimations.py`, `tests/test_schemas.py`);
- el endpoint SSE `/api/v1/estimate/stream` (eventos `token`/`done`/`error` con `prompt_version`) y `GET /api/v1/context`;
- los templates de prompt sin tocar el LLM: la descripción dentro de `<project_description>`, el condicional de `output_format`, el de `detail_level`, la inclusión de ejemplos y `StrictUndefined` (`tests/prompts/test_estimation_v1.py`);
- el cliente HTTP del frontend con `httpx.MockTransport` (parseo SSE, métricas, `estimate()` y mapeo de errores) y que `frontend/` no importa `app.*` (`tests/test_frontend_client.py`, `tests/test_frontend_decoupling.py`);
- la detección de truncamiento, de respuesta vacía y de errores del proveedor (incluido que el detalle interno no se filtra al cliente);
- la caché de respuestas (clave determinista, TTL, memoria y Redis con `fakeredis`) y el wrapper de LiteLLM (normalización, fallback, coste y cacheo) con el `Router` mockeado;
- el logging estructurado (eventos, coste, cache_hit y que las claves no se filtran) y la propagación de `X-Request-ID`;
- la derivación del modelo, el arranque sin credenciales y la validación automática de la estructura de carpetas (`tests/test_project_structure.py`).

## Calidad y CI

El proyecto usa `ruff` (lint + formato) y `mypy` (type-check estricto sobre `app/`):

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy app frontend
```

El pipeline de GitHub Actions (`.github/workflows/ci.yml`) se dispara en `push` a `main`/`feature/**` y en pull requests, y ejecuta tres jobs:

1. **quality** — `ruff` + `mypy`.
2. **test** — `pytest` (incluye la validación de estructura).
3. **docker** — build, tests dentro de la imagen y smoke test de `/health` levantando el servicio con Compose.

## Docker

Requisitos: Docker con Compose. Las claves se inyectan en runtime desde `.env` (nunca quedan en la imagen).

```bash
# 1. Configurar variables de entorno
cp .env.example .env
# Edita .env y añade tus claves (OPEN_AI_KEY / ANTHROPIC_API_KEY)

# 2. Construir y arrancar
docker compose up --build -d

# 3. Comprobar
curl http://localhost:8000/health

# Ver logs
docker compose logs -f api

# Parar
docker compose down
```

- Swagger: http://localhost:8000/docs
- Interfaz Streamlit: http://localhost:8501 (servicio `ui`, misma imagen que la API; habla con el servicio `api` por la red interna vía `API_BASE_URL=http://api:8000`)
- Redis: servicio `redis` (caché persistente de respuestas; la API lo usa por la red interna con `CACHE_BACKEND=redis`)
- Puerto personalizado: `API_PORT=8123 docker compose up --build -d` (y `UI_PORT=8502` para Streamlit)
- Arrancar solo la interfaz: `docker compose up --build ui`
- Tests dentro de Docker:

```bash
docker compose --profile test run --rm test
```

La imagen es multi-stage: `runtime` (imagen final mínima con uvicorn), `test` (dev deps + pytest) y `builder` (resolución de dependencias con uv).

## Variables de entorno

| Variable                  | Descripción                                       | Default       |
| ------------------------- | ------------------------------------------------- | ------------- |
| `APP_ENV`                 | Entorno de ejecución                              | `development` |
| `LOG_LEVEL`               | Nivel de logging (`DEBUG`, `INFO`, ...)           | `DEBUG`       |
| `LOG_FORMAT`              | Formato de logs: `console` o `json`               | `console`     |
| `LLM_PROVIDER`            | Proveedor activo: `openai`, `anthropic` o `custom` | `openai`      |
| `LLM_MODEL`               | Modelo del proveedor activo                       | default del proveedor (`gpt-4o-mini` para `openai`) |
| `LLM_FALLBACK_MODEL`      | Modelo de fallback si el primario falla (vacío = sin fallback) | — |
| `LLM_ROUTING_MODE`        | Router de LiteLLM: `fallback` (primario estricto) o `balanced` (reparto) | `fallback` |
| `TEMPERATURE`             | Temperatura de generación (no aplica a `anthropic`) | `0.2`       |
| `LLM_TIMEOUT_SECONDS`     | Timeout de la llamada al proveedor (segundos)     | `30`          |
| `LLM_MAX_RETRIES`         | Reintentos del cliente del SDK                    | `2`           |
| `LLM_MAX_TOKENS`          | Máximo de tokens de salida                        | `2048`        |
| `CACHE_BACKEND`           | Caché de respuestas: `memory`, `redis` o `none`   | `memory`      |
| `CACHE_TTL`               | TTL de las entradas de caché (segundos)           | `86400`       |
| `REDIS_URL`               | URL de Redis (cuando `CACHE_BACKEND=redis`)       | `redis://localhost:6379` |
| `DESCRIPTION_MIN_LENGTH`  | Longitud mínima de la descripción (caracteres)    | `20`          |
| `DESCRIPTION_MAX_LENGTH`  | Longitud máxima de la descripción (caracteres)    | `50000`       |
| `OPEN_AI_KEY`             | API key de OpenAI                                 | —             |
| `ANTHROPIC_API_KEY`       | API key de Anthropic                              | —             |
| `CUSTOM_LLM_BASE_URL`     | URL base del endpoint OpenAI-compatible           | —             |
| `CUSTOM_LLM_API_KEY`      | API key del endpoint OpenAI-compatible            | —             |
| `API_BASE_URL`            | URL base de la API que consume el frontend        | `http://localhost:8000` |

> En Docker Compose, el servicio `api` fuerza `CACHE_BACKEND=redis` y `REDIS_URL=redis://redis:6379` (servicio `redis` de la misma red).

> Si `LLM_MODEL` se deja vacío, se deriva del proveedor activo. El proveedor `custom` no tiene default: exige `LLM_MODEL` explícito.
>
> `TEMPERATURE` no aplica al proveedor `anthropic` (su SDK no acepta el parámetro). Si se configura un valor distinto del default, el servicio emite un warning `llm.temperature.ignored` la primera vez.

## Proveedor custom (OpenAI-compatible)

Además de `openai` y `anthropic`, el proyecto admite un proveedor `custom` que habla el formato de la API de OpenAI. Permite apuntar a cualquier servidor compatible (NaN API, Ollama, LM Studio, vLLM, OpenRouter, Groq, Together…) indicando la URL base, la API key y el modelo desde `.env`:

```dotenv
LLM_PROVIDER=custom
LLM_MODEL=qwen3.8-flash
CUSTOM_LLM_BASE_URL=https://api.nan.builders/v1
CUSTOM_LLM_API_KEY=tu-api-key
```

> El proveedor `custom` usa el endpoint **Chat Completions** (`/v1/chat/completions`), no la Responses API, porque la mayoría de servidores compatibles con OpenAI solo implementan el primero. Recuerda que, a diferencia de `openai` y `anthropic`, este proveedor **no tiene modelo por defecto**: `LLM_MODEL` es obligatorio.
>
> En modelos de razonamiento (p. ej. `qwen3.8-flash`), el proveedor puede devolver un `reasoning_content` además del `content`. La aplicación ignora el razonamiento y solo muestra la estimación final (`choices[0].message.content` / `delta.content`).
>
> En streaming, el proveedor custom solicita el uso de tokens con `stream_options={"include_usage": True}`, de modo que la interfaz muestra los tokens de entrada/salida también en streaming. Si un endpoint compatible no soportara esa opción, se mostraría `—` en las métricas.
