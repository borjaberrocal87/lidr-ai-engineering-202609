# estimador-cag

API FastAPI que recibe la transcripción de una reunión y devuelve una estimación de software generada por un LLM, usando **arquitectura CAG** (Context-Augmented Generation): el contexto estático (ejemplos de estimaciones previas) se inyecta directamente en el prompt en cada llamada. Sin base de datos, sin retrieval, sin persistencia.

## Estructura

```
estimador-cag/
├── app/
│   ├── main.py              # App FastAPI, router /api/v1, GET /health, /docs
│   ├── config.py            # Settings (Pydantic BaseSettings) desde .env
│   ├── routers/
│   │   └── estimations.py   # POST /api/v1/estimate + schemas Pydantic
│   ├── services/
│   │   └── llm_service.py   # System prompt + ejemplos + llamada al proveedor
│   └── context/
│       └── examples.py      # Estimaciones previas (few-shot, CAG)
├── tests/                   # Tests con pytest (proveedores mockeados)
├── Dockerfile               # Build multi-stage (builder / test / runtime)
├── docker-compose.yml       # Servicios api y test
├── .dockerignore
├── .env.example
├── pyproject.toml
└── README.md
```

## Requisitos

- Python 3.11+
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
  "input_tokens": 1234,
  "output_tokens": 567
}
```

Estado del servicio:

```bash
curl http://localhost:8000/health
```

## Tests

```bash
uv run pytest
```

Los tests mockean los proveedores LLM (no hacen llamadas reales) y cubren el endpoint, la validación de schemas y la inyección del contexto CAG en el system prompt.

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
- Puerto personalizado: `API_PORT=8123 docker compose up --build -d`
- Tests dentro de Docker:

```bash
docker compose --profile test run --rm test
```

La imagen es multi-stage: `runtime` (imagen final mínima con uvicorn), `test` (dev deps + pytest) y `builder` (resolución de dependencias con uv).

## Variables de entorno

| Variable            | Descripción                                   | Default             |
| ------------------- | --------------------------------------------- | ------------------- |
| `APP_ENV`           | Entorno de ejecución                          | `development`       |
| `LOG_LEVEL`         | Nivel de logging (`DEBUG`, `INFO`, ...)       | `DEBUG`             |
| `LLM_PROVIDER`      | Proveedor activo: `openai` o `anthropic`      | `openai`            |
| `LLM_MODEL`         | Modelo del proveedor activo                   | `gpt-4o-mini`       |
| `OPEN_AI_KEY`       | API key de OpenAI                             | —                   |
| `ANTHROPIC_API_KEY` | API key de Anthropic                          | —                   |
