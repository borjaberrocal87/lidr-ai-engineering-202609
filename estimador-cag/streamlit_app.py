"""Interfaz conversacional (Streamlit) del estimador de software CAG.

Reutiliza la lógica de llamada al LLM del backend FastAPI: pega la
transcripción de una reunión y obtén la estimación generada en streaming.
"""

import time

import streamlit as st

from app.services.llm_service import (
    LLMConfigurationError,
    StreamMetrics,
    stream_estimation,
)

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
