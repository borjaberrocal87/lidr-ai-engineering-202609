"""Interfaz conversacional (Streamlit) del estimador de software CAG.

Reutiliza la lógica de llamada al LLM del backend FastAPI: pega la
transcripción de una reunión y obtén la estimación generada.
"""

import streamlit as st

from app.services.llm_service import LLMConfigurationError, generate_estimation

st.set_page_config(
    page_title="Estimador de software CAG",
    page_icon="🧮",
    layout="centered",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

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
        try:
            result = generate_estimation(prompt)
        except LLMConfigurationError as exc:
            st.error(str(exc))
        else:
            st.markdown(result.estimation)
            st.session_state.messages.append({"role": "assistant", "content": result.estimation})
            st.caption(f"Modelo: {result.model} · Proveedor: {result.provider}")
