# Spec — Proyecto 1: Proveedor LLM custom compatible con OpenAI

## Objetivo

Añadir al Proyecto 1 un tercer proveedor LLM de tipo **custom** que habla el formato de la API de OpenAI (OpenAI-compatible).

A diferencia de `openai` y `anthropic`, cuyas credenciales son fijas, este proveedor se configura por completo desde `.env`:

- La **URL base** del endpoint (Ollama, LM Studio, vLLM, OpenRouter, Groq, Together…)
- La **API key**
- El **modelo**

Al finalizar este ejercicio, tendrás un estimador capaz de funcionar contra cualquier servidor compatible con OpenAI sin tocar código: basta con cambiar `LLM_PROVIDER=custom`, la URL base, la key y el modelo en `.env`.

## Contexto del proyecto

Partimos del proyecto de la sesión 03 (backend FastAPI + interfaz Streamlit). La lógica de estimación ya es agnóstica del transporte: recibe una transcripción y devuelve una estimación.

La selección de proveedor vive en `app/services/llm_service.py`: `generate_estimation()` y `stream_estimation()` deciden con un `if/elif` sobre `settings.llm_provider` (ver `llm_service.py:79` y `llm_service.py:100`). Añadir un proveedor consiste en tres cosas:

1. Extender el `Literal` de `LLM_PROVIDER` y añadir sus variables en `app/config.py`.
2. Implementar sus variantes no-streaming y streaming en `llm_service.py`.
3. Reflejarlo en `.env.example`, en el README y en los tests.

> **Nota importante sobre la API:** el proveedor `openai` del proyecto usa la **Responses API** (`client.responses.create`, endpoint `/responses`; ver `llm_service.py:191`). La mayoría de servidores OpenAI-compatible **no implementan `/responses`**: solo exponen **Chat Completions** (`/v1/chat/completions`). Por eso el proveedor `custom` debe usar `client.chat.completions.create(...)` y **no** reutilizar `_estimate_with_openai`.

No hace falta añadir dependencias: el SDK `openai` ya está en `pyproject.toml` y su parámetro `base_url` es justo lo que permite apuntar a cualquier servidor compatible.

## Requisitos para el ejercicio

- Proyecto de la sesión 02/03 funcionando.
- Un endpoint OpenAI-compatible accesible (local o remoto) con su URL base, API key y nombre de modelo. Ejemplos:
  - NaN API: `https://api.nan.builders/v1` (modelos tipo `qwen3.8-flash`, `deepseek-v4-flash`…)
  - Ollama: `http://localhost:11434/v1`
  - LM Studio: `http://localhost:1234/v1`
  - vLLM: `http://localhost:8000/v1`
  - OpenRouter: `https://openrouter.ai/api/v1`
  - Groq: `https://api.groq.com/openai/v1`
- SDK `openai` (ya presente en el proyecto como dependencia de la Sesión 02).

## ✍ Ejercicio

### Paso 1 — Configuración con variables de entorno

Amplía `Settings` en `app/config.py` para aceptar el nuevo proveedor y sus credenciales:

```python
llm_provider: Literal["openai", "anthropic", "custom"] = "openai"
llm_model: str = DEFAULT_MODELS["openai"]
temperature: float = 0.2

open_ai_key: str = ""
anthropic_api_key: str = ""

# Proveedor custom (OpenAI-compatible)
custom_llm_base_url: str = ""
custom_llm_api_key: str = ""
```

Añade las nuevas variables a `.env.example`:

```dotenv
# Proveedor LLM activo: openai | anthropic | custom
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
TEMPERATURE=0.2

OPEN_AI_KEY=
ANTHROPIC_API_KEY=

# Proveedor custom (OpenAI-compatible)
CUSTOM_LLM_BASE_URL=
CUSTOM_LLM_API_KEY=
```

> **Sobre el modelo:** `LLM_MODEL` se reutiliza para los tres proveedores. A diferencia de `openai` y `anthropic`, el proveedor `custom` **no tiene modelo por defecto**: es obligatorio indicarlo en `.env`. Si activas `custom`, no olvides ajustar también `LLM_MODEL`.

### Paso 2 — Llamada no-streaming

En `services/llm_service.py`, añade la rama del nuevo proveedor en `generate_estimation`:

```python
if settings.llm_provider == "custom":
    return _estimate_with_openai_compatible(transcription)
```

E implementa la función. Fíjate en las diferencias respecto a `_estimate_with_anthropic`:

- El system prompt viaja como mensaje `system` dentro de `messages`, no como parámetro `system`.
- El texto de la respuesta está en `response.choices[0].message.content`.
- Los tokens se llaman `prompt_tokens` / `completion_tokens` (no `input_tokens` / `output_tokens`).
- `base_url` es obligatorio para apuntar al servidor compatible.

```python
def _estimate_with_openai_compatible(transcription: str) -> EstimationResult:
    from openai import OpenAI

    client = OpenAI(
        api_key=_require_custom_key(),
        base_url=_require_custom_base_url(),
    )
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": transcription},
        ],
        temperature=settings.temperature,
    )

    usage = response.usage
    return EstimationResult(
        estimation=response.choices[0].message.content or "",
        model=settings.llm_model,
        provider="custom",
        temperature=settings.temperature,
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
    )
```

### Paso 3 — Streaming

Añade la rama en `stream_estimation` (validando la configuración de forma anticipada, como ya hacen los otros proveedores):

```python
if settings.llm_provider == "custom":
    _require_custom_base_url()
    _require_custom_key()
    return _stream_with_openai_compatible(transcription, metrics)
```

E implementa el generador usando `stream=True`:

```python
def _stream_with_openai_compatible(
    transcription: str,
    metrics: StreamMetrics | None,
) -> Iterator[str]:
    from openai import OpenAI

    client = OpenAI(
        api_key=_require_custom_key(),
        base_url=_require_custom_base_url(),
    )
    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": transcription},
        ],
        temperature=settings.temperature,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

    _record_metrics(
        metrics,
        model=settings.llm_model,
        provider="custom",
        usage=None,
    )
```

> **Sobre el `usage` en streaming:** solo llega si el servidor lo soporta (con `stream_options={"include_usage": True}` y el último chunk). Trata la métrica como **opcional**: si no hay datos de usage, los campos de tokens deben quedar a `None`. El modelo y el proveedor sí se pueden rellenar siempre.

### Paso 4 — Validación de configuración

Añade dos helpers junto a `_require_openai_key` y `_require_anthropic_key`, con mensajes claros y sin valores hardcodeados:

```python
def _require_custom_base_url() -> str:
    if not settings.custom_llm_base_url:
        raise LLMConfigurationError(
            "CUSTOM_LLM_BASE_URL no está configurada. Añádela al archivo .env."
        )
    return settings.custom_llm_base_url


def _require_custom_key() -> str:
    if not settings.custom_llm_api_key:
        raise LLMConfigurationError(
            "CUSTOM_LLM_API_KEY no está configurada. Añádela al archivo .env."
        )
    return settings.custom_llm_api_key
```

El endpoint `POST /api/v1/estimate` ya traduce `LLMConfigurationError` a un `503 Service Unavailable` con el detalle, así que no hay que tocar el router.

### Paso 5 — Tests

Añade tests en `tests/test_llm_service.py` siguiendo el patrón de mockeo existente (`monkeypatch` sobre `openai.OpenAI`, `SimpleNamespace` para las respuestas):

- `test_custom_missing_base_url_raises` — sin URL base lanza `LLMConfigurationError` con `CUSTOM_LLM_BASE_URL`.
- `test_custom_missing_key_raises` — sin API key lanza `LLMConfigurationError` con `CUSTOM_LLM_API_KEY`.
- `test_custom_provider_maps_response` — mockea `chat.completions.create` y comprueba el `EstimationResult` (`provider="custom"`, tokens `prompt_tokens`/`completion_tokens`).
- Verifica que la `base_url`, la key y el `model` llegan al cliente falso.
- Verifica que el system prompt enviado contiene los ejemplos CAG (`ESTIMATION_EXAMPLES[0]["estimation"] in ...`).
- `test_custom_stream_yields_deltas_and_metrics` — mockea el generador de chunks (`delta.content`) y valida el streaming, incluyendo el caso de un chunk sin `choices`.

> El `Literal` de `LLM_PROVIDER` ahora admite `custom`, así que revisa cualquier test que asuma solo dos proveedores válidos.

### Paso 6 — Documentación

- Añade las nuevas variables a `.env.example` (Paso 1).
- Actualiza la tabla de variables de entorno del `README.md`: `LLM_PROVIDER` admite `custom`, y añade filas para `CUSTOM_LLM_BASE_URL` y `CUSTOM_LLM_API_KEY`.
- Documenta en el README que el proveedor `custom` usa el endpoint Chat Completions (`/v1/chat/completions`) y ofrece un ejemplo de configuración.

### Paso 7 — Verificación

Configura un servidor compatible (por ejemplo la NaN API) y arranca el proyecto:

```dotenv
# .env
LLM_PROVIDER=custom
LLM_MODEL=qwen3.8-flash
CUSTOM_LLM_BASE_URL=https://api.nan.builders/v1
CUSTOM_LLM_API_KEY=tu-api-key
```

```bash
# Backend
uv run uvicorn app.main:app --reload

# Endpoint
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "El cliente necesita una landing page con formulario de contacto e integración con HubSpot."
  }'

# Interfaz conversacional
uv run streamlit run streamlit_app.py
```

Comprueba que la respuesta indica `"provider": "custom"` y que el streaming funciona en Streamlit.

## Checklist de verificación

Antes de considerar el ejercicio completado, verifica:

- [ ] `LLM_PROVIDER=custom` funciona tanto en `POST /api/v1/estimate` como en Streamlit
- [ ] `LLM_PROVIDER=openai` y `LLM_PROVIDER=anthropic` siguen funcionando exactamente igual
- [ ] La URL base, la API key y el modelo se leen de `.env` (nunca hardcodeados)
- [ ] El proveedor `custom` usa `/chat/completions` (no `/responses`)
- [ ] Si falta la URL base o la API key, el endpoint devuelve un error claro (503)
- [ ] El system prompt CAG (con los ejemplos) sigue inyectándose en todas las llamadas
- [ ] `uv run pytest`, `uv run ruff check .` y `uv run mypy app` pasan

## Documentación de referencia

- SDK de OpenAI (Python) — parámetro `base_url` del cliente: https://github.com/openai/openai-python
- Compatibilidad OpenAI en Ollama: https://github.com/ollama/ollama/blob/main/docs/openai.md
- LM Studio — servidor compatible con OpenAI: https://lmstudio.ai/docs/local-server
- vLLM — OpenAI-compatible server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- OpenRouter API: https://openrouter.ai/docs

## Nota

A medida que crezca el número de proveedores, el `if/elif` de `llm_service.py` debería evolucionar hacia un wrapper/registry de proveedores (una interfaz común con una implementación por proveedor). Es la evolución natural que ya se apuntó como opcional al cierre de la sesión 03, pero queda fuera del alcance de este ejercicio.
