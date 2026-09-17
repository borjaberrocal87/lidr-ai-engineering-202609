"""Interfaz conversacional (Streamlit) del estimador de software CAG.

Reutiliza la lógica de llamada al LLM del backend FastAPI: pega la
transcripción de una reunión y obtén la estimación generada en streaming.
"""

import time

import streamlit as st

from app.context.examples import ESTIMATION_EXAMPLES
from app.logging_config import configure_logging
from app.services.llm_service import (
    LLMConfigurationError,
    StreamMetrics,
    build_system_prompt,
    stream_estimation,
)

configure_logging()

st.set_page_config(
    page_title="Estimador de software CAG",
    page_icon="🧮",
    layout="centered",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

st.title("Estimador de software")
st.caption(
    "Pega la transcripción de una reunión con un cliente y recibirás una "
    "estimación de software generada por un LLM con arquitectura CAG."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pega aquí la transcripción de la reunión..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        metrics = StreamMetrics(model="", provider="")
        started_at = time.perf_counter()
        try:
            full_response = st.write_stream(stream_estimation(prompt, metrics))
        except LLMConfigurationError as exc:
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
            }
            st.caption(f"Modelo: {metrics.model} · Proveedor: {metrics.provider}")

with st.sidebar:
    st.header("Contexto CAG")

    st.subheader("System prompt activo")
    st.code(build_system_prompt(), language="markdown")

    st.subheader("Contexto inyectado")
    st.caption(f"{len(ESTIMATION_EXAMPLES)} estimaciones de referencia en el prompt")
    for index, example in enumerate(ESTIMATION_EXAMPLES, start=1):
        with st.expander(f"Ejemplo {index}"):
            st.markdown(f"**Resumen de la reunión:**\n\n{example['meeting_summary']}")
            st.markdown(example["estimation"])

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
            f"- **Tiempo de respuesta:** {elapsed:.2f} s"
        )
