# Spec — Proyecto 1: Interfaz conversacional con Streamlit

> **Actualización — arquitectura por capas (refactor posterior).**
> El proyecto evolucionó para desacoplar la UI del backend. Donde este spec
> habla de reutilizar módulos Python del backend (`build_system_prompt()`,
> `ESTIMATION_EXAMPLES`, `stream_estimation`) o de `streamlit_app.py` en la
> raíz, léelo como:
>
> - La UI vive en `frontend/streamlit_app.py` y consume la API **por HTTP**
>   mediante `frontend/client.py` (no importa `app.*`).
> - El system prompt, los ejemplos CAG y los límites se obtienen de
>   `GET /api/v1/context`.
> - El streaming se obtiene de `POST /api/v1/estimate/stream` (SSE).
> - Arranque: `uv run streamlit run frontend/streamlit_app.py` (con la API
>   levantada y `API_BASE_URL` apuntando a ella).
>
> Así se puede sustituir Streamlit por otra UI reutilizando el mismo cliente.

## Objetivo

Añadir una interfaz conversacional web al Proyecto 1 usando Streamlit.

Al finalizar este ejercicio, tendrás una aplicación que:

- Muestra una interfaz de chat en el navegador para pegar transcripciones de reunión
- Reutiliza la lógica de llamada al LLM del backend CAG (sesión 02)
- Muestra la estimación generada en streaming (token a token)
- Mantiene el historial de la conversación visible durante la sesión

El objetivo es no depender de `curl`, Postman ni Swagger para probar el estimador.

## Contexto del proyecto

Partimos del proyecto de la sesión 02: un backend FastAPI con un endpoint CAG (`POST /api/v1/estimate`) que recibe transcripciones y devuelve estimaciones de software.

La lógica ya existe y se reutiliza tal cual:

- `app/services/llm_service.py` — construye el system prompt, inyecta el contexto estático (CAG) y llama al proveedor.
- `app/context/examples.py` — ejemplos de estimaciones previas que alimentan el contexto.
- `app/config.py` — configuración y API keys cargadas desde `.env`.

Esta capa es una interfaz adicional: no sustituye a la API, la **consume por HTTP** a través de `frontend/client.py`. El frontend no importa `app.*` (arquitectura por capas).

## Requisitos para el ejercicio

- Python 3.11+ instalado (el proyecto usa 3.12)
- `uv` instalado como gestor de paquetes
- API key del proveedor (OpenAI o Anthropic) ya configurada en `.env` desde la sesión 01/02

## ✍ Ejercicio

### Paso 1 — Añadir Streamlit al proyecto

Añade Streamlit como dependencia:

```bash
uv add streamlit
```

Crea el fichero `frontend/streamlit_app.py` (capa de presentación). Se ejecutará con:

```bash
uv run streamlit run frontend/streamlit_app.py
```

Estructura orientativa (ya con la separación por capas):

```
estimador-cag/
├── app/                          # Backend (API) — dueño del LLM y las claves
│   ├── services/llm_service.py
│   ├── context/examples.py
│   ├── schemas/estimations.py
│   └── config.py
├── frontend/                     # Capa de presentación — consume la API por HTTP
│   ├── client.py                 # ← cliente HTTP (httpx)
│   ├── config.py                 # API_BASE_URL
│   └── streamlit_app.py          # ← nuevo
├── .env
└── .streamlit/
    └── secrets.toml              # opcional, alternativa a .env
```

> **Nota:** `frontend/streamlit_app.py` es un punto de entrada independiente del servidor FastAPI. No importa `app.*`: habla con la API por HTTP usando `frontend/client.py`.

### Paso 2 — Chat básico (Nivel 1, obligatorio)

Crea una aplicación Streamlit con interfaz de chat usando `st.chat_message` y `st.chat_input`.

Requisitos:

1. El usuario debe poder **escribir o pegar** una transcripción de reunión.
2. La aplicación envía ese texto a la API (`POST /api/v1/estimate` o `POST /api/v1/estimate/stream`) a través de `frontend/client.py`.
3. La estimación resultante se muestra como **mensaje del asistente**.
4. El historial de la conversación debe mantenerse visible durante la sesión. Usa `st.session_state` para guardarlo y repintarlo en cada rerun de Streamlit.
5. El system prompt debe ser el mismo que usa el endpoint CAG — no lo dupliques, obtenlo de `GET /api/v1/context`.
6. La API key **no debe estar hardcodeada**.

Patrón orientativo:

```python
import streamlit as st

# Inicializar historial una sola vez por sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

# Repintar historial en cada rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
if prompt := st.chat_input("Pega aquí la transcripción de la reunión..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # ... llamada al LLM y render de la respuesta ...
```

### Paso 3 — Streaming (Nivel 2, obligatorio)

Modifica la aplicación para que la respuesta del LLM se muestre en **streaming** (token a token) en lugar de aparecer de golpe al terminar la generación.

Requisitos:

- Usa `st.write_stream` o el patrón de `placeholder` + delta que prefieras.
- El usuario debe ver la estimación "escribiéndose" en tiempo real.
- Para OpenAI: usa `client.responses.stream(...)` o `stream=True` y expón un generador de texto.
- Para Anthropic: usa `client.messages.stream(...)` y expón un generador de texto.
- El historial debe seguir guardando la respuesta **completa** una vez terminado el streaming (el generador se consume una sola vez).

Patrón orientativo:

```python
with st.chat_message("assistant"):
    full_response = st.write_stream(stream_estimation(prompt))
st.session_state.messages.append({"role": "assistant", "content": full_response})
```

> **Nota:** si reutilizas `generate_estimation`, esta devuelve la respuesta completa. Para el nivel 2 necesitarás una variante con streaming en `llm_service.py` (por ejemplo `stream_estimation`) que exponga un generador. El system prompt debe seguir siendo `build_system_prompt()`.

### Paso 4 — Contexto CAG en la interfaz (Nivel 3, opcional)

Añade un panel lateral con `st.sidebar` que muestre:

1. **System prompt activo** (solo lectura) — `system_prompt` de `GET /api/v1/context`.
2. **Contexto estático inyectado** — los `examples` de `GET /api/v1/context`.
3. **Métricas básicas de la última llamada**: modelo utilizado, tokens de entrada, tokens de salida y tiempo de respuesta.

Esto da al usuario visibilidad sobre qué información está usando el modelo para generar la estimación.

### Paso 5 — Verificación

Arranca la aplicación y pruébala en el navegador:

```bash
uv run streamlit run frontend/streamlit_app.py
```

Pega una transcripción de ejemplo (puedes usar `examples/transcripcion.md`) y comprueba que la estimación aparece en streaming y que la conversación persiste al hacer varias preguntas seguidas.

## Checklist de verificación

Antes de considerar el ejercicio completado, verifica:

- [ ] `streamlit run streamlit_app.py` abre una interfaz de chat en el navegador
- [ ] Puedes pegar una transcripción de reunión y recibes una estimación de software
- [ ] La conversación persiste en pantalla (puedes hacer varias preguntas seguidas)
- [ ] La respuesta se muestra en streaming, no de golpe
- [ ] La API key se lee desde `.env` o `st.secrets`, no está en el código

## Documentación de referencia

- Streamlit chat elements: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
- SDK de tu proveedor (OpenAI o Anthropic): documentación de streaming
- Streamlit secrets management: https://docs.streamlit.io/develop/concepts/connections/secrets-management

## Nota

El wrapper de abstracción de proveedores, el cacheo inteligente de respuestas y la capa de logging/trazabilidad los implementaremos como opcional.
