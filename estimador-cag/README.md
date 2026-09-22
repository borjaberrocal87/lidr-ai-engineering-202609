# estimador-cag

API FastAPI que recibe la transcripción de una reunión y devuelve una estimación de software generada por un LLM, usando **arquitectura CAG** (Context-Augmented Generation): el contexto estático (ejemplos de estimaciones previas) se inyecta directamente en el prompt en cada llamada. Sin base de datos, sin retrieval, sin persistencia.

## Alcance

El contexto CAG es un catálogo en memoria (`app/context/examples.py`), sin base de datos ni retrieval. Es una decisión, no deuda pendiente.

El umbral para revisarla: cuando el catálogo crezca más allá de unos 15-20 ejemplos, toca migrar a una fuente vectorial (RAG). El punto de cambio está aislado en `build_system_prompt(examples=...)`, de modo que el router y el servicio no se enteran.

## Estructura

```
estimador-cag/
├── app/
│   ├── main.py              # App FastAPI, router /api/v1, GET /health, /docs
│   ├── config.py            # Settings (Pydantic BaseSettings) desde .env
│   ├── logging_config.py    # Configuración de structlog (console/json)
│   ├── routers/
│   │   └── estimations.py   # POST /api/v1/estimate + schemas Pydantic
│   ├── services/
│   │   └── llm_service.py   # System prompt + ejemplos + llamada al proveedor
│   └── context/
│       └── examples.py      # Estimaciones previas (few-shot, CAG)
├── streamlit_app.py         # Interfaz conversacional web (Streamlit)
├── tests/                   # Tests con pytest (proveedores mockeados + estructura)
├── examples/
│   └── transcripcion.md     # Transcripción de reunión de ejemplo (input del ejercicio)
├── specs/
│   ├── sesion-2-scaffolding-fastapi.md  # Spec del backend FastAPI (sesión 2)
│   ├── sesion-3-interfaz-conversacional-streamlit.md  # Spec de la UI (sesión 3)
│   └── proveedor-openai-compatible.md  # Spec del proveedor custom (OpenAI-compatible)
├── Dockerfile               # Build multi-stage (builder / test / runtime)
├── docker-compose.yml       # Servicios api y test
├── .dockerignore
├── .env.example
├── pyproject.toml
└── README.md
```

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

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "En la reunión con el equipo de marketing, el cliente explicó que necesita una landing page con formulario de contacto, integración con su CRM actual (HubSpot), y una sección de blog con editor WYSIWYG. El plazo ideal sería tenerlo listo en 4 semanas. El diseño ya existe en Figma."
  }'
```

Respuesta:

```json
{
  "estimation": "## Estimación: ...",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "temperature": 0.2,
  "input_tokens": 1234,
  "output_tokens": 567,
  "truncated": false
}
```

`truncated: true` significa que el modelo agotó `LLM_MAX_TOKENS` antes de terminar y la estimación puede estar incompleta. En ese caso la API responde igualmente `200` (para no perder la respuesta parcial), pero lo indica de forma explícita y deja un warning en los logs.

Errores:

- `503` — falta la configuración del proveedor activo (p. ej. la API key). El mensaje nombra la variable de entorno.
- `502` — el proveedor LLM falló (red, timeout, rate limit o estado). El detalle interno se registra en el servidor y **no** viaja al cliente.
- `500` — error inesperado en el código.

Estado del servicio:

```bash
curl http://localhost:8000/health
```

`/health` responde siempre `200` (no depende del LLM) e incluye `llm_configured` para saber si el proveedor activo tiene credenciales.

## Límites de la llamada al LLM

La llamada al proveedor está acotada por configuración:

- `LLM_TIMEOUT_SECONDS` (por defecto `30`) y `LLM_MAX_RETRIES` (por defecto `2`) se aplican al cliente del SDK, para no heredar el timeout por defecto de diez minutos.
- `LLM_MAX_TOKENS` (por defecto `2048`) acota el coste de salida y se traduce en `truncated` cuando la respuesta se corta.
- `TRANSCRIPTION_MIN_LENGTH` / `TRANSCRIPTION_MAX_LENGTH` (por defecto `10` / `50000` caracteres) se validan en el borde (422) antes de gastar un token. El servicio repite la comprobación por si lo invoca otro adaptador (worker, CLI); en ese caso responde `422`.

La transcripción se envuelve en una etiqueta con un sufijo aleatorio por petición (`<transcripcion-XXXX>...</transcripcion-XXXX>`) y el system prompt explica al modelo que ese bloque son datos, no instrucciones. Como el nombre de la etiqueta no es previsible, el texto de entrada no puede cerrarla.

## Interfaz conversacional (Streamlit)

Además de la API, el proyecto incluye una interfaz de chat web para pegar transcripciones y ver la estimación en streaming, sin usar `curl`, Postman ni Swagger:

```bash
uv run streamlit run streamlit_app.py
```

Se abre en http://localhost:8501. La conversación persiste durante la sesión (`st.session_state`) y la app reutiliza la misma lógica de llamada y el mismo system prompt CAG que el endpoint `/api/v1/estimate`. La API key se sigue leyendo desde `.env`.

El panel lateral (nivel 3) muestra el system prompt activo en solo lectura, los ejemplos de contexto CAG inyectados y las métricas de la última llamada: modelo, proveedor, tokens de entrada/salida y tiempo de respuesta.

## Logging

Las llamadas al LLM se registran con [structlog](https://www.structlog.org/) (integrado con `logging`), tanto desde la API como desde Streamlit. Cada llamada emite:

- `llm.call.start` / `llm.stream.start` — proveedor, modelo, temperatura y longitud de la transcripción.
- `llm.call.end` / `llm.stream.end` — tokens de entrada/salida, si la respuesta se truncó y latencia en milisegundos.
- `llm.call.error` / `llm.stream.error` — error con traceback y latencia. La versión en streaming añade `llm.stream.aborted` si el cliente corta antes de terminar.
- `llm.response.truncated` (en el router) — warning cuando el modelo agotó `LLM_MAX_TOKENS`.

```text
2026-09-17T15:04:54.945815Z [info] llm.call.start provider=custom model=qwen3.8-flash temperature=0.2 transcription_chars=842
2026-09-17T15:05:03.123456Z [info] llm.call.end   provider=custom model=qwen3.8-flash input_tokens=1234 output_tokens=567 latency_ms=8421.3
```

Por privacidad se registran **solo metadatos**: nunca la API key ni el texto de la transcripción (posible información confidencial del cliente). El nivel se controla con `LOG_LEVEL` y el formato con `LOG_FORMAT` (`console` para texto legible, `json` para agregadores). En Docker los logs salen por stdout y se consultan con `docker compose logs -f api` (o `ui`).

## Transcripción de ejemplo

En `examples/transcripcion.md` hay una transcripción de reunión realista (landing page + integración HubSpot + blog con editor WYSIWYG) lista para usar como parámetro del ejercicio. Copia el contenido de la sección **Transcripción** en el campo `transcription` del body.

## Tests

```bash
uv run pytest
```

Los tests mockean los proveedores LLM (no hacen llamadas reales) y cubren:

- el endpoint `/api/v1/estimate` y los schemas de entrada/salida (incluida la validación de longitud mínima/máxima y que una entrada inválida **no** llega a invocar al LLM);
- la inyección del contexto CAG en el system prompt y el delimitado de la transcripción con nonce;
- la detección de truncamiento, de respuesta vacía y de errores del proveedor (incluido que el detalle interno no se filtra al cliente);
- la derivación del modelo, el arranque sin credenciales y la validación automática de la estructura de carpetas (`tests/test_project_structure.py`).

## Calidad y CI

El proyecto usa `ruff` (lint + formato) y `mypy` (type-check estricto sobre `app/`):

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy app
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
- Interfaz Streamlit: http://localhost:8501 (servicio `ui`, misma imagen que la API)
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
| `TEMPERATURE`             | Temperatura de generación (no aplica a `anthropic`) | `0.2`       |
| `LLM_TIMEOUT_SECONDS`     | Timeout de la llamada al proveedor (segundos)     | `30`          |
| `LLM_MAX_RETRIES`         | Reintentos del cliente del SDK                    | `2`           |
| `LLM_MAX_TOKENS`          | Máximo de tokens de salida                        | `2048`        |
| `TRANSCRIPTION_MIN_LENGTH` | Longitud mínima de la transcripción (caracteres) | `10`          |
| `TRANSCRIPTION_MAX_LENGTH` | Longitud máxima de la transcripción (caracteres) | `50000`       |
| `OPEN_AI_KEY`             | API key de OpenAI                                 | —             |
| `ANTHROPIC_API_KEY`       | API key de Anthropic                              | —             |
| `CUSTOM_LLM_BASE_URL`     | URL base del endpoint OpenAI-compatible           | —             |
| `CUSTOM_LLM_API_KEY`      | API key del endpoint OpenAI-compatible            | —             |

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
