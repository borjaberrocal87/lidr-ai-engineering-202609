"""Interfaz conversacional (Streamlit) del estimador de software CAG.

Capa de presentación: consume la API FastAPI por HTTP a través de
`frontend.client`. No importa `app.*`, de modo que se puede sustituir por otra
UI reutilizando el mismo cliente.
"""

import sys
from pathlib import Path

# `streamlit run frontend/streamlit_app.py` pone en sys.path el directorio del
# script (`frontend/`), no la raíz del proyecto, así que `import frontend` no
# resuelve. Añadimos la raíz explícitamente (local y Docker).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import time  # noqa: E402

import streamlit as st  # noqa: E402

from frontend.client import (  # noqa: E402
    ApiConfigurationError,
    ApiError,
    ApiInputError,
    ApiProviderError,
    ApiUnavailableError,
    get_context,
    stream_estimation,
)
from frontend.logging_config import configure_logging  # noqa: E402
from frontend.models import ContextResponse, StreamMetrics  # noqa: E402

configure_logging()

st.set_page_config(
    page_title="Estimador de software CAG",
    page_icon="🧮",
    layout="centered",
)


@st.cache_data(ttl=300, show_spinner=False)
def _load_context() -> ContextResponse:
    return get_context()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

st.title("Estimador de software")
st.caption(
    "Pega la transcripción de una reunión con un cliente y recibirás una "
    "estimación de software generada por un LLM con arquitectura CAG."
)

try:
    context = _load_context()
except ApiUnavailableError as exc:
    st.error(f"{exc} Arranca la API o revisa API_BASE_URL.")
    st.stop()
except ApiError as exc:
    st.error(f"No se pudo cargar el contexto de la API: {exc}")
    st.stop()

if not context.llm_configured:
    st.warning(
        "El proveedor LLM no está configurado en la API. Revisa las claves en "
        "el `.env` del servidor."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pega aquí la transcripción de la reunión..."):
    min_length = context.transcription_min_length
    max_length = context.transcription_max_length
    if not min_length <= len(prompt) <= max_length:
        st.error(
            f"La transcripción debe tener entre {min_length} y "
            f"{max_length} caracteres (tiene {len(prompt)})."
        )
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            metrics = StreamMetrics()
            started_at = time.perf_counter()
            try:
                full_response = st.write_stream(stream_estimation(prompt, metrics))
            except ApiConfigurationError as exc:
                st.error(str(exc))
            except ApiInputError as exc:
                st.error(str(exc))
            except ApiProviderError:
                st.error("No se pudo generar la estimación. Inténtalo de nuevo más tarde.")
            except ApiUnavailableError:
                st.error("Se perdió la conexión con la API. Inténtalo de nuevo.")
            except ApiError as exc:
                st.error(str(exc))
            else:
                elapsed_seconds = time.perf_counter() - started_at
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.last_metrics = {
                    "model": metrics.model,
                    "provider": metrics.provider,
                    "input_tokens": metrics.input_tokens,
                    "output_tokens": metrics.output_tokens,
                    "elapsed_seconds": elapsed_seconds,
                    "truncated": metrics.truncated,
                }
                if metrics.truncated:
                    st.warning(
                        "La estimación se cortó al alcanzar el límite de tokens "
                        "(`LLM_MAX_TOKENS`): puede estar incompleta."
                    )
                st.caption(f"Modelo: {metrics.model} · Proveedor: {metrics.provider}")

with st.sidebar:
    st.header("Contexto CAG")

    st.subheader("System prompt activo")
    st.code(context.system_prompt, language="markdown")

    st.subheader("Contexto inyectado")
    st.caption(f"{len(context.examples)} estimaciones de referencia en el prompt")
    for index, example in enumerate(context.examples, start=1):
        with st.expander(f"Ejemplo {index}"):
            st.markdown(f"**Resumen de la reunión:**\n\n{example.meeting_summary}")
            st.markdown(example.estimation)

    st.subheader("Última llamada")
    last_metrics = st.session_state.last_metrics
    if last_metrics is None:
        st.caption("Todavía no se ha generado ninguna estimación.")
    else:
        elapsed = last_metrics["elapsed_seconds"]
        input_tokens = last_metrics["input_tokens"]
        output_tokens = last_metrics["output_tokens"]
        st.markdown(
            f"- **Modelo:** {last_metrics['model']}\n"
            f"- **Proveedor:** {last_metrics['provider']}\n"
            f"- **Tokens de entrada:** {input_tokens if input_tokens is not None else '—'}\n"
            f"- **Tokens de salida:** {output_tokens if output_tokens is not None else '—'}\n"
            f"- **Tiempo de respuesta:** {elapsed:.2f} s\n"
            f"- **Truncada:** {'sí' if last_metrics.get('truncated') else 'no'}"
        )
