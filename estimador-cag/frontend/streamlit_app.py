"""UI conversacional (Streamlit) del estimador de software.

Capa de presentación: consume la API FastAPI por HTTP a través de
`frontend.client`. No importa `app.*`.

A diferencia de la sesión 04 (un formulario transaccional), aquí hay una
**sesión conversacional**: se crea al cargar la página, el `session_id` vive en
`st.session_state` y cada turno acepta una transcripción más adjuntos PDF/DOCX.
El panel lateral muestra la `project_metadata` (memoria) separada del historial.
"""

import sys
from pathlib import Path

# `streamlit run frontend/streamlit_app.py` pone en sys.path el directorio del
# script (`frontend/`), no la raíz, así que `import frontend` no resuelve.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from frontend.client import (  # noqa: E402
    ApiConfigurationError,
    ApiError,
    ApiGuardrailError,
    ApiInputError,
    ApiNotFoundError,
    ApiProviderError,
    ApiUnavailableError,
    create_session,
    estimate_in_session,
    get_context,
)
from frontend.logging_config import configure_logging  # noqa: E402
from frontend.models import (  # noqa: E402
    DETAIL_LEVELS,
    OUTPUT_FORMATS,
    PROJECT_TYPES,
    ContextResponse,
    ProjectMetadata,
    SessionEstimateResponse,
)

configure_logging()

st.set_page_config(
    page_title="Estimador de software CAG",
    page_icon="🧮",
    layout="wide",
)

_ATTACHMENT_MIME = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@st.cache_data(ttl=300, show_spinner=False)
def _load_context() -> ContextResponse:
    return get_context()


def _ensure_session() -> str:
    """Crea una sesión si no hay ninguna en el estado. Si la API la perdió, la recrea."""
    session_id = st.session_state.get("session_id")
    if session_id:
        return str(session_id)
    created = create_session()
    st.session_state.session_id = created
    return created


def _new_conversation() -> None:
    st.session_state.session_id = create_session()
    st.session_state.turns = []


def _render_estimation(response: SessionEstimateResponse) -> None:
    est = response.result
    st.markdown(f"**Versión del prompt:** `{response.prompt_version}`")
    if est.out_of_scope:
        st.warning(est.summary)
        return
    st.markdown(est.summary)
    duration_col, cost_col, confidence_col = st.columns(3)
    duration_col.metric("Duración", f"{est.total_duration_weeks} sem")
    cost_col.metric("Coste", f"{est.total_cost_eur:,} €")
    confidence_col.metric("Confianza", f"{est.confidence_pct}%")
    table = ["| Fase | Semanas | Coste (EUR) | Detalle |", "|---|---:|---:|---|"]
    table.extend(
        f"| {phase.name} | {phase.duration_weeks} | {phase.cost_eur:,} | {phase.summary} |"
        for phase in est.phases
    )
    st.markdown("\n".join(table))
    if response.fallback_used:
        st.caption("Se usó el modelo de fallback.")
    st.caption(f"Modelo: {response.model} · Proveedor: {response.provider}")


if "turns" not in st.session_state:
    st.session_state.turns = []

st.title("Estimador de software — sesión conversacional")
st.caption(
    "Crea una sesión, describe el proyecto y refina el alcance turno a turno. "
    "Puedes adjuntar PDFs o Word para enriquecer la estimación."
)

try:
    context = _load_context()
except ApiUnavailableError as exc:
    st.error(f"{exc} Arranca la API o revisa API_BASE_URL.")
    st.stop()
except ApiError as exc:
    st.error(f"No se pudo cargar el contexto de la API: {exc}")
    st.stop()

try:
    session_id = _ensure_session()
except ApiUnavailableError as exc:
    st.error(f"No se pudo crear la sesión: {exc}")
    st.stop()

# Historial de turnos.
for index, turn in enumerate(st.session_state.turns, start=1):
    st.subheader(f"Turno {index}")
    st.markdown(f"> {turn['transcript']}")
    if turn.get("attachments"):
        st.caption("Adjuntos: " + ", ".join(turn["attachments"]))
    _render_estimation(turn["response"])
    st.divider()

st.subheader("Nuevo turno")
with st.form("turn_form", clear_on_submit=True):
    transcript = st.text_area(
        "Transcripción / descripción del turno",
        height=160,
        placeholder="Añade información, refina el alcance o corrige supuestos…",
        help=(
            f"Entre {context.description_min_length} y {context.description_max_length} caracteres."
        ),
    )
    uploads = st.file_uploader(
        "Adjuntos (PDF o DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )
    col_type, col_detail, col_format = st.columns(3)
    project_type = col_type.selectbox("Tipo de proyecto", options=PROJECT_TYPES, index=1)
    detail_level = col_detail.radio(
        "Nivel de detalle", options=DETAIL_LEVELS, index=1, horizontal=True
    )
    output_format = col_format.selectbox("Formato de salida", options=OUTPUT_FORMATS, index=0)
    submitted = st.form_submit_button("Añadir turno", type="primary")

if submitted:
    cleaned = transcript.strip()
    if len(cleaned) < context.description_min_length:
        st.error(
            f"La transcripción debe tener al menos {context.description_min_length} caracteres."
        )
    else:
        attachments = [
            (
                upload.name,
                upload.getvalue(),
                _ATTACHMENT_MIME.get(Path(upload.name).suffix.lower(), "application/octet-stream"),
            )
            for upload in (uploads or [])
            if upload is not None
        ]
        with st.spinner("Generando estimación del turno…"):
            try:
                response = estimate_in_session(
                    session_id,
                    transcript=cleaned,
                    project_type=project_type,
                    detail_level=detail_level,
                    output_format=output_format,
                    attachments=attachments,
                )
            except ApiNotFoundError:
                # La API se reinició y perdió la sesión en memoria: recreamos.
                st.session_state.session_id = create_session()
                st.warning(
                    "La sesión había caducado; se ha creado una nueva. Vuelve a enviar el turno."
                )
            except ApiGuardrailError as exc:
                st.warning(f"Transcripción rechazada por los guardrails: {exc.message}")
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
                st.session_state.turns.append(
                    {
                        "transcript": cleaned,
                        "attachments": [upload.name for upload in (uploads or [])],
                        "response": response,
                    }
                )
                st.session_state.last_metrics = {
                    "prompt_version": response.prompt_version,
                    "model": response.model,
                    "provider": response.provider,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "fallback_used": response.fallback_used,
                    "history_messages": response.history_messages,
                }
                st.rerun()

with st.sidebar:
    st.header("Sesión")
    st.code(session_id, language="text")
    if st.button("Nueva conversación", use_container_width=True):
        _new_conversation()
        st.rerun()

    metadata: ProjectMetadata = (
        st.session_state.turns[-1]["response"].metadata
        if st.session_state.turns
        else ProjectMetadata()
    )
    st.subheader("Memoria del proyecto")
    st.caption("Hechos preservados entre turnos, separados del historial.")
    technologies = ", ".join(metadata.mentioned_technologies) or "—"
    st.markdown(
        f"- **Nombre:** {metadata.project_name or '—'}\n"
        f"- **Equipo:** {metadata.assumed_team_size or '—'}\n"
        f"- **Tecnologías:** {technologies}\n"
        f"- **Alcance:** {metadata.agreed_scope or '—'}"
    )

    st.subheader("Historial")
    history_messages = (
        st.session_state.turns[-1]["response"].history_messages if st.session_state.turns else 0
    )
    st.markdown(
        f"- **Turnos:** {len(st.session_state.turns)}\n"
        f"- **Mensajes en ventana:** {history_messages}"
    )

    last_metrics = st.session_state.get("last_metrics")
    if last_metrics:
        cost = last_metrics["cost_usd"]
        cost_label = f"{cost:.4f} USD" if cost is not None else "—"
        st.subheader("Última llamada")
        st.markdown(
            f"- **Prompt:** `{last_metrics['prompt_version']}`\n"
            f"- **Modelo:** {last_metrics['model']}\n"
            f"- **Proveedor:** {last_metrics['provider']}\n"
            f"- **Tokens (in/out):** {last_metrics['input_tokens']} / "
            f"{last_metrics['output_tokens']}\n"
            f"- **Coste:** {cost_label}\n"
            f"- **Fallback:** {'sí' if last_metrics['fallback_used'] else 'no'}"
        )

    st.subheader("Prompt activo")
    st.caption("Renderizado con la configuración por defecto (sin metadata de sesión).")
    st.code(context.system_prompt, language="markdown")
