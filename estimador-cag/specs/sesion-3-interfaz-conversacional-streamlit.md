# Spec — Proyecto 1: Interfaz conversacional con Streamlit

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

Esta capa es una interfaz adicional: no sustituye a la API, la consume desde dentro del propio proyecto Python.

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

Crea el fichero `streamlit_app.py` en la **raíz** del proyecto. Se ejecutará con:

```bash
uv run streamlit run streamlit_app.py
```

Estructura orientativa:

```
estimador-cag/
├── app/
│   ├── services/llm_service.py   # (reutilizado)
│   ├── context/examples.py       # (reutilizado)
│   └── config.py                 # (reutilizado)
├── streamlit_app.py              # ← nuevo
├── .env
└── .streamlit/
    └── secrets.toml              # opcional, alternativa a .env
```

> **Nota:** `streamlit_app.py` vive en la raíz y no dentro de `app/`, porque es un punto de entrada independiente del servidor FastAPI. Puede importar `app.*` sin problema.

### Paso 2 — Chat básico (Nivel 1, obligatorio)

Crea una aplicación Streamlit con interfaz de chat usando `st.chat_message` y `st.chat_input`.

Requisitos:

1. El usuario debe poder **escribir o pegar** una transcripción de reunión.
2. La aplicación envía ese texto al LLM reutilizando la lógica que ya tienes (`generate_estimation` de `app/services/llm_service.py`).
3. La estimación resultante se muestra como **mensaje del asistente**.
4. El historial de la conversación debe mantenerse visible durante la sesión. Usa `st.session_state` para guardarlo y repintarlo en cada rerun de Streamlit.
5. El system prompt debe ser el mismo que usa el endpoint CAG — no lo dupliques, importa `build_system_prompt()`.
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

1. **System prompt activo** (solo lectura) — `build_system_prompt()`.
2. **Contexto estático inyectado** — los ejemplos de `ESTIMATION_EXAMPLES`.
3. **Métricas básicas de la última llamada**: modelo utilizado, tokens de entrada, tokens de salida y tiempo de respuesta.

Esto da al usuario visibilidad sobre qué información está usando el modelo para generar la estimación.

### Paso 5 — Verificación

Arranca la aplicación y pruébala en el navegador:

```bash
uv run streamlit run streamlit_app.py
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
