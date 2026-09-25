"""Formulario (Streamlit) del estimador de software.

Capa de presentación: consume la API FastAPI por HTTP a través de
`frontend.client`. No importa `app.*`, de modo que se puede sustituir por otra
UI reutilizando el mismo cliente.

El usuario ya no escribe en un chat: rellena un formulario que produce un
`EstimationRequest` tipado (descripción + tipo de proyecto + nivel de detalle +
formato de salida) y lo envía a `POST /api/v1/estimate`.
"""

import sys
from pathlib import Path

# `streamlit run frontend/streamlit_app.py` pone en sys.path el directorio del
# script (`frontend/`), no la raíz del proyecto, así que `import frontend` no
# resuelve. Añadimos la raíz explícitamente (local y Docker).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from frontend.client import (  # noqa: E402
    ApiConfigurationError,
    ApiError,
    ApiInputError,
    ApiProviderError,
    ApiUnavailableError,
    estimate,
    get_context,
)
from frontend.logging_config import configure_logging  # noqa: E402
from frontend.models import (  # noqa: E402
    DETAIL_LEVELS,
    OUTPUT_FORMATS,
    PROJECT_TYPES,
    ContextResponse,
)

configure_logging()

st.set_page_config(
    page_title="Estimador de software CAG",
    page_icon="🧮",
    layout="centered",
)


@st.cache_data(ttl=300, show_spinner=False)
def _load_context() -> ContextResponse:
    return get_context()


if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

st.title("Estimador de software")
st.caption(
    "Rellena el formulario y recibirás una estimación de software generada por "
    "un LLM con arquitectura CAG, siguiendo el formato que elijas."
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

with st.form("estimation_form", clear_on_submit=False):
    description = st.text_area(
        "Descripción del proyecto",
        height=200,
        placeholder="Describe objetivos, funcionalidades clave y restricciones…",
        help=(
            f"Entre {context.description_min_length} y {context.description_max_length} caracteres."
        ),
    )
    project_type = st.selectbox("Tipo de proyecto", options=PROJECT_TYPES, index=1)
    detail_level = st.radio(
        "Nivel de detalle",
        options=DETAIL_LEVELS,
        index=1,
        horizontal=True,
    )
    output_format = st.selectbox("Formato de salida", options=OUTPUT_FORMATS, index=0)
    versions = context.available_versions or [context.prompt_version or "v1"]
    default_version = context.prompt_version if context.prompt_version in versions else versions[0]
    prompt_version = st.selectbox(
        "Versión del prompt",
        options=versions,
        index=versions.index(default_version),
    )
    submitted = st.form_submit_button("Generar estimación", type="primary")

if submitted:
    cleaned = description.strip()
    if len(cleaned) < context.description_min_length:
        st.error(f"La descripción debe tener al menos {context.description_min_length} caracteres.")
    else:
        payload = {
            "description": cleaned,
            "project_type": project_type,
            "detail_level": detail_level,
            "output_format": output_format,
        }
        with st.spinner("Llamando al estimador…"):
            try:
                result = estimate(payload, prompt_version=prompt_version)
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
                st.session_state.last_metrics = {
                    "prompt_version": result.prompt_version,
                    "model": result.model,
                    "provider": result.provider,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "cache_hit": result.cache_hit,
                    "cost_usd": result.cost_usd,
                    "fallback_used": result.fallback_used,
                    "truncated": result.truncated,
                }
                st.markdown(f"**Versión del prompt:** `{result.prompt_version}`")
                st.markdown(result.estimation)
                if result.truncated:
                    st.warning(
                        "La estimación se cortó al alcanzar el límite de tokens "
                        "(`LLM_MAX_TOKENS`): puede estar incompleta."
                    )
                if result.cache_hit:
                    st.caption("Respuesta servida desde caché (cache hit).")
                if result.fallback_used:
                    st.caption("Se usó el modelo de fallback.")
                st.caption(f"Modelo: {result.model} · Proveedor: {result.provider}")

with st.sidebar:
    st.header("Prompt activo")
    st.caption("Se muestra el prompt renderizado con la configuración por defecto.")
    st.code(context.system_prompt, language="markdown")

    st.subheader("Límites")
    st.markdown(
        f"- **Descripción mínima:** {context.description_min_length} caracteres\n"
        f"- **Descripción máxima:** {context.description_max_length} caracteres"
    )

    st.subheader("Última llamada")
    last_metrics = st.session_state.last_metrics
    if last_metrics is None:
        st.caption("Todavía no se ha generado ninguna estimación.")
    else:
        input_tokens = last_metrics["input_tokens"]
        output_tokens = last_metrics["output_tokens"]
        cost_usd = last_metrics.get("cost_usd")
        st.markdown(
            f"- **Prompt:** `{last_metrics.get('prompt_version', '—')}`\n"
            f"- **Modelo:** {last_metrics['model']}\n"
            f"- **Proveedor:** {last_metrics['provider']}\n"
            f"- **Tokens de entrada:** {input_tokens if input_tokens is not None else '—'}\n"
            f"- **Tokens de salida:** {output_tokens if output_tokens is not None else '—'}\n"
            f"- **Coste estimado:** {f'{cost_usd:.4f} USD' if cost_usd is not None else '—'}\n"
            f"- **Desde caché:** {'sí' if last_metrics.get('cache_hit') else 'no'}\n"
            f"- **Fallback:** {'sí' if last_metrics.get('fallback_used') else 'no'}\n"
            f"- **Truncada:** {'sí' if last_metrics.get('truncated') else 'no'}"
        )
