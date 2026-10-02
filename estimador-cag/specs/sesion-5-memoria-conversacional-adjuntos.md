# Spec — Proyecto 1: Memoria conversacional y adjuntos

## Objetivo

Hasta la sesión 04 el estimator es un sistema transaccional: una descripción entra, una estimación sale. Funciona para "estimar este proyecto y olvidarme", pero no modela lo que de verdad pasa en una empresa: una conversación iterativa donde el cliente refina el alcance, añade información, sube documentos complementarios y espera que el sistema recuerde sobre qué proyecto está hablando.

Al terminar este ejercicio, el estimator:

- Mantiene **memoria conversacional** dentro de una sesión: el contexto del proyecto en curso se preserva entre turnos sin reenviar el historial bruto completo.
- Separa **historial** (el array `messages` que viaja al modelo) de **memoria** (los hechos del proyecto, `project_metadata`).
- Acepta **adjuntos** PDF y Word que enriquecen la transcripción.
- Expone un endpoint multi-turno `POST /api/v1/sessions/{session_id}/estimate` que actualiza historial y metadata en cada llamada.
- Aplica una **ventana deslizante** de N turnos con el system prompt como invariante.

> **Reconciliación con el repo.** El punto de partida de la sesión 04 ya está implementado en `main`: salida estructurada (`EstimationResult` con validadores), guardrails de entrada/salida, caché exacta + semántica, prompts Jinja2 versionados (v1/v2), clave de caché canónica y formulario tipado en el cliente. Esta sesión **se apoya** en todo eso y añade las capas conversacionales por encima. El endpoint de formulario `/api/v1/estimate` no se toca; el endpoint con sesión, deliberadamente, **no cachea** (cada turno depende del historial y de la metadata).

## Contexto del proyecto

Arrancamos del estado actual de `main`. Piezas relevantes que se reutilizan o extienden:

- `app/services/llm_wrapper.py` — wrapper LiteLLM. Ya expone `complete_structured(system_prompt, user_message, response_model, …) -> (modelo, meta)` con reintento ante `ValidationError`. **Le falta** una variante que acepte un array `messages` (system + historial + user): la añadimos.
- `app/services/llm_service.py` — orquesta guardrails, caché y llamada. `generate_structured_estimation` es el pipeline del formulario. Añadiremos el pipeline conversacional aquí.
- `app/prompts/loader.py` — `render_estimation_prompt(request, version)` con `StrictUndefined`, `trim_blocks`/`lstrip_blocks`, huella y logging. Le añadiremos `metadata` al contexto y un render para el extractor.
- `app/schemas/estimations.py` — `EstimationRequest`, `EstimationResult` (validado), `StructuredEstimateResponse`, `Phase`.
- `app/guardrails/` — `check_input` (entrada) y `enforce_scope_response` (salida).
- `app/dependencies.py` — `get_llm_wrapper`, `get_cache`, `get_semantic_cache`, `get_openai_client`. Añadiremos `get_session_store`.
- `frontend/` — capa de presentación que habla HTTP por `frontend/client.py` y no importa `app.*`.

## Requisitos para el ejercicio

- Proyecto de la sesión 04 en verde (`uv run pytest`, `uv run ruff check .`, `uv run mypy app frontend`).
- Dependencias nuevas:
  - `python-multipart` (FastAPI lo exige para `Form`/`File`).
  - `pypdf` y `python-docx` (extracción local de adjuntos, Camino B).
  - `pytest-asyncio` (tests de integración con `httpx.AsyncClient`).
- Añadir `asyncio_mode = "auto"` a `[tool.pytest.ini_options]`.
- API key configurada para poder probar de verdad (los tests mockean el LLM).

```bash
uv add python-multipart pypdf python-docx
uv add --dev pytest-asyncio
```

## ✍ Ejercicio

### Paso 1 — Modelar el estado de la sesión

Crea `app/sessions/` con dos estructuras y la sesión que las agrupa. Sin BBDD, sin Redis: un diccionario en memoria del proceso.

**`app/sessions/models.py`**

```python
"""Estado en memoria de una sesión conversacional.

Diseño:
- El **system prompt no forma parte del historial**: se regenera en cada turno
  a partir de la `ProjectMetadata` actual (que evoluciona). La invariante es que
  toda llamada al modelo lleva system + ventana de historial + turno actual.
- `ConversationHistory` recorta por pares (user+assistant) para que la
  alternancia de roles no se rompa nunca.
- `ProjectMetadata` mezcla de forma aditiva: escalares se sobrescriben, la lista
  de tecnologías se une (case-insensitive) para que un stack nuevo no borre el
  anterior.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationHistory(BaseModel):
    """Ventana deslizante de turnos (pares user+assistant)."""

    max_turns: int = Field(default=6, ge=1)
    messages: list[Message] = Field(default_factory=list)

    def append(self, *, user: str, assistant: str) -> None:
        self.messages.append(Message(role="user", content=user))
        self.messages.append(Message(role="assistant", content=assistant))
        self._trim()

    def to_messages_list(self, system_prompt: str) -> list[dict[str, str]]:
        """Array `messages` listo para el modelo, con el system regenerado delante.

        El system se recibe por parámetro (no se almacena) porque depende de la
        metadata actual: guardarlo en el historial congelaría una versión vieja.
        """
        return [
            {"role": "system", "content": system_prompt},
            *({"role": m.role, "content": m.content} for m in self.messages),
        ]

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        overflow = len(self.messages) - max_messages
        if overflow > 0:
            if overflow % 2 != 0:  # nunca dejar un par partido
                overflow += 1
            del self.messages[:overflow]


class ProjectMetadata(BaseModel):
    """Hechos del proyecto que se preservan entre turnos.

    Todos los campos son opcionales: en el primer turno todavía no hay nada
    comprometido. El extractor los rellena turno a turno.
    """

    project_name: str | None = Field(default=None, max_length=120)
    assumed_team_size: int | None = Field(default=None, ge=1, le=50)
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = Field(default=None, max_length=2000)

    def is_empty(self) -> bool:
        return (
            self.project_name is None
            and self.assumed_team_size is None
            and not self.mentioned_technologies
            and self.agreed_scope is None
        )

    def merge_with(self, update: "ProjectMetadata") -> "ProjectMetadata":
        """Escalares: gana el valor no nulo de `update`. Tecnologías: unión."""
        merged_tech = list(self.mentioned_technologies)
        seen = {t.lower() for t in merged_tech}
        for tech in update.mentioned_technologies:
            if tech.lower() not in seen:
                merged_tech.append(tech)
                seen.add(tech.lower())
        return ProjectMetadata(
            project_name=update.project_name or self.project_name,
            assumed_team_size=update.assumed_team_size or self.assumed_team_size,
            mentioned_technologies=merged_tech,
            agreed_scope=update.agreed_scope or self.agreed_scope,
        )


class Session(BaseModel):
    """Sesión conversacional. Vive en memoria del proceso.

    Aceptamos la volatilidad en esta fase: reiniciar el servicio borra la
    memoria y, con más de un worker, cada worker tendría su propia copia. La
    persistencia entra en el módulo de RAG.
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    history: ConversationHistory = Field(default_factory=ConversationHistory)
    metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

**`app/sessions/store.py`**

```python
"""Almacén de sesiones en memoria, indexado por session_id.

La clase existe para aislar el punto de sustitución (Redis, Postgres…) sin
tocar el router ni el servicio. No es thread-safe: `uvicorn --workers=1` es el
escenario soportado.
"""

from __future__ import annotations

from app.sessions.models import ConversationHistory, Session


class SessionNotFoundError(KeyError):
    """Se lanza cuando el id no existe."""


class SessionStore:
    def __init__(self, *, max_turns: int = 6) -> None:
        self._sessions: dict[str, Session] = {}
        self._max_turns = max_turns

    def create(self) -> Session:
        session = Session(history=ConversationHistory(max_turns=self._max_turns))
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_404(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(session_id) from exc

    def __len__(self) -> int:
        return len(self._sessions)
```

En `app/dependencies.py`, añade el singleton:

```python
@lru_cache
def get_session_store() -> SessionStore:
    """Store en memoria, uno por worker. Ver nota de volatilidad en el módulo."""
    return SessionStore(max_turns=settings.max_conversation_turns)
```

### Paso 2 — Endpoint para crear sesiones

Crea `app/routers/sessions.py` con el alta y una vista de debug:

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_session_store
from app.sessions.models import ProjectMetadata
from app.sessions.store import SessionNotFoundError, SessionStore

router = APIRouter(prefix="/sessions", tags=["sessions"])


class CreateSessionResponse(BaseModel):
    session_id: str = Field(description="UUID de la nueva sesión conversacional.")


class SessionInfoResponse(BaseModel):
    session_id: str
    message_count: int
    max_turns: int
    metadata: ProjectMetadata


@router.post("", response_model=CreateSessionResponse, status_code=201)
def create_session(store: SessionStore = Depends(get_session_store)) -> CreateSessionResponse:
    session = store.create()
    return CreateSessionResponse(session_id=session.session_id)


@router.get("/{session_id}", response_model=SessionInfoResponse)
def get_session(
    session_id: str, store: SessionStore = Depends(get_session_store)
) -> SessionInfoResponse:
    try:
        session = store.get_or_404(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc
    return SessionInfoResponse(
        session_id=session.session_id,
        message_count=len(session.history.messages),
        max_turns=session.history.max_turns,
        metadata=session.metadata,
    )
```

Registra el router en `app/main.py`:

```python
from app.routers import estimations, sessions

app.include_router(estimations.router, prefix="/api/v1", tags=["estimations"])
app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])
```

### Paso 3 — Adjuntos (Camino B: extracción local)

Elegimos **extraer el texto en el servicio** con `pypdf`/`python-docx` y concatenarlo a la transcripción. Ventajas: es independiente del proveedor, funciona igual con OpenAI o Anthropic, y deja el texto listo para el chunking de RAG del módulo 3. El precio es que perdemos la comprensión multimodal (diagramas) que daría el Camino A.

**`app/attachments/extractor.py`**

```python
"""Extracción local de texto de adjuntos PDF/DOCX (Camino B).

El router es el que tiene los `UploadFile`; aquí solo entra (filename, bytes).
El texto se trunca a `max_chars` para proteger el presupuesto del prompt.
"""

from __future__ import annotations

import io
from pathlib import PurePosixPath

import structlog

log = structlog.get_logger()

_SUPPORTED_EXTS = {".pdf", ".docx"}


class AttachmentExtractionError(Exception):
    def __init__(self, filename: str, message: str) -> None:
        super().__init__(message)
        self.filename = filename
        self.message = message


class UnsupportedAttachmentError(AttachmentExtractionError):
    """Extensión no soportada."""


def _extension(filename: str) -> str:
    return PurePosixPath(filename).suffix.lower()


def _extract_pdf(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    parts: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 — una página rota no tumba el resto
            log.warning("pdf_page_extract_failed", error=str(exc)[:200])
            text = ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _extract_docx(content: bytes) -> str:
    from docx import Document

    document = Document(io.BytesIO(content))
    return "\n".join(p.text for p in document.paragraphs if p.text and p.text.strip())


def extract_text(*, filename: str, content: bytes, max_chars: int) -> str:
    ext = _extension(filename)
    if ext not in _SUPPORTED_EXTS:
        raise UnsupportedAttachmentError(
            filename, f"Extensión no soportada: {ext!r}; soportadas: {sorted(_SUPPORTED_EXTS)}"
        )
    try:
        text = _extract_pdf(content) if ext == ".pdf" else _extract_docx(content)
    except AttachmentExtractionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AttachmentExtractionError(filename, f"No se pudo extraer {filename}: {exc}") from exc

    text = text.strip()
    if len(text) > max_chars:
        log.info("attachment_truncated", filename=filename, original_chars=len(text))
        text = text[:max_chars]
    return text


def enrich_transcript(*, transcript: str, attachments: list[tuple[str, str]]) -> str:
    """Concatena la transcripción con cada adjunto entre fences explícitos."""
    if not attachments:
        return transcript
    parts = [transcript.strip()]
    for filename, text in attachments:
        if text:
            parts.append(f"--- attachment: {filename} ---\n{text}\n--- end attachment ---")
    return "\n\n".join(parts)
```

**Endpoint multi-turno** (en `app/routers/sessions.py`): `multipart/form-data` con `transcript`, los enums tipados como `Form` y la lista opcional de ficheros.

```python
from fastapi import Depends, File, Form, HTTPException, UploadFile

from app.attachments.extractor import (
    AttachmentExtractionError,
    UnsupportedAttachmentError,
    enrich_transcript,
    extract_text,
)
from app.config import settings
from app.dependencies import get_session_store
from app.guardrails.input import InputGuardrailViolation
from app.routers.estimations import validated_prompt_version
from app.schemas.estimations import (
    DetailLevel,
    OutputFormat,
    ProjectType,
    SessionEstimateResponse,
)
from app.services.llm_service import (
    LLMConfigurationError,
    LLMProviderError,
    generate_session_estimation,
)
from app.sessions.store import SessionNotFoundError, SessionStore


@router.post("/{session_id}/estimate", response_model=SessionEstimateResponse)
async def estimate_in_session(
    session_id: str,
    transcript: str = Form(
        ..., min_length=settings.description_min_length, max_length=settings.description_max_length
    ),
    project_type: ProjectType = Form(...),
    detail_level: DetailLevel = Form(...),
    output_format: OutputFormat = Form(...),
    version: str = Depends(validated_prompt_version),
    attachments: list[UploadFile] = File(default_factory=list),
    store: SessionStore = Depends(get_session_store),
) -> SessionEstimateResponse:
    try:
        session = store.get_or_404(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="session_not_found") from exc

    extracted: list[tuple[str, str]] = []
    for upload in attachments or []:
        if not upload.filename:
            continue
        content = await upload.read()
        try:
            text = extract_text(
                filename=upload.filename, content=content, max_chars=settings.max_attachment_chars
            )
        except UnsupportedAttachmentError as exc:
            raise HTTPException(
                status_code=415,
                detail={"reason": "unsupported_attachment", "filename": exc.filename},
            ) from exc
        except AttachmentExtractionError as exc:
            raise HTTPException(
                status_code=422,
                detail={"reason": "attachment_extraction_failed", "filename": exc.filename},
            ) from exc
        if text:
            extracted.append((upload.filename, text))

    enriched = enrich_transcript(transcript=transcript, attachments=extracted)
    try:
        outcome = generate_session_estimation(
            session=session,
            transcript=enriched,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            version=version,
        )
    except InputGuardrailViolation as exc:
        raise HTTPException(
            status_code=400, detail={"reason": exc.reason, "message": exc.message}
        ) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=502, detail="No se pudo generar la estimación.") from exc

    return SessionEstimateResponse(
        session_id=session.session_id,
        result=outcome.result,
        prompt_version=version,
        metadata=session.metadata,
        history_messages=len(session.history.messages),
        model=outcome.model,
        provider=outcome.provider,
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
        cost_usd=outcome.cost_usd,
        fallback_used=outcome.fallback_used,
    )
```

> **Nota sobre longitudes.** El campo `transcript` se valida con los límites de descripción (`description_min_length`/`description_max_length`). El texto **enriquecido** con adjuntos puede superarlos, así que el servicio no vuelve a aplicar el guard de longitud sobre él; cada adjunto se acota por separado con `max_attachment_chars`. Los guardrails de entrada sí se ejecutan sobre el texto enriquecido.

### Paso 4 — `project_metadata` en el prompt y extracción

**Bloque en el system prompt.** Añade al final de `app/prompts/estimation/v1/system.j2` y `v2/system.j2` un bloque condicional (vacío en el primer turno):

```jinja
{% if metadata and not metadata_is_empty %}
<project_metadata>
Hechos ya establecidos sobre el proyecto en turnos anteriores. Trátalos como fuente de verdad: si la transcripción nueva contradice este bloque, gana el bloque salvo que el usuario esté corrigiendo información previa.
- project_name: {{ metadata.project_name or "desconocido" }}
- assumed_team_size: {{ metadata.assumed_team_size or "desconocido" }}
- mentioned_technologies: {% if metadata.mentioned_technologies %}{{ metadata.mentioned_technologies | join(", ") }}{% else %}ninguna{% endif %}
- agreed_scope: {{ metadata.agreed_scope or "sin acordar" }}
</project_metadata>
{% endif %}
```

y a `<reglas>` (solo en la práctica: no rompe el formulario porque el bloque no se renderiza si no hay metadata):

- Trata el historial de la conversación como contexto ya establecido; no pidas al usuario que repita lo que ya está en `<project_metadata>`.
- Si la transcripción añade alcance a un proyecto ya estimado, produce una estimación completa actualizada, no un delta.

**Loader.** Extiende `render_estimation_prompt` para aceptar metadata opcional y añade el render del extractor:

```python
from app.sessions.models import ProjectMetadata


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = DEFAULT_ESTIMATION_PROMPT_VERSION,
    metadata: ProjectMetadata | None = None,
) -> tuple[str, str]:
    context = {
        "description": request.description,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "reference_projects": [p.model_dump() for p in (request.reference_projects or [])],
        "metadata": metadata,
        "metadata_is_empty": metadata is None or metadata.is_empty(),
    }
    ...


def render_metadata_extraction_prompt(
    *,
    transcript: str,
    result: EstimationResult,
    previous: ProjectMetadata,
    version: str = "v1",
) -> tuple[str, str]:
    context = {
        "transcript": transcript,
        "result": result,
        "phases": result.phases,
        "previous": previous,
        "previous_is_empty": previous.is_empty(),
    }
    system = _env.get_template(f"metadata_extraction/{version}/system.j2").render(**context)
    user = _env.get_template(f"metadata_extraction/{version}/user.j2").render(**context)
    return system, user
```

**`app/prompts/metadata_extraction/v1/system.j2`**

```jinja
Eres un extractor de hechos de proyecto. Lees el último turno de una conversación de estimación y devuelves un objeto ProjectMetadata con los hechos durables del proyecto.

Esquema:
- project_name (string | null, máx 120): el nombre propio que dio el usuario. null si no lo ha nombrado.
- assumed_team_size (int | null, 1-50): tamaño de equipo que asume la estimación. null si se desconoce.
- mentioned_technologies (lista[string]): solo las tecnologías NUEVAS de este turno que no estén ya en la lista previa.
- agreed_scope (string | null, máx 2000): un párrafo con el alcance acordado. Actualízalo cuando el usuario añada o quite alcance.

Reglas estrictas:
- Devuelve null (o lista vacía) antes que inventar. Si el turno no menciona el tamaño de equipo, deja assumed_team_size en null: el merge conserva el valor anterior.
- No inventes hechos que no estén en la transcripción o en el resumen de la estimación.
- Normaliza la capitalización de tecnologías ("Postgres", "React", "AWS Lambda").
- El merge aguas arriba sobrescribe escalares con valores no nulos y une tecnologías; devolver un dato incorrecto SÍ sobrescribe uno correcto. Ante la duda, null.
```

**`app/prompts/metadata_extraction/v1/user.j2`**

```jinja
<previous_metadata>
{% if previous_is_empty %}
(empty — this is the first turn)
{% else %}
- project_name: {{ previous.project_name or "null" }}
- assumed_team_size: {{ previous.assumed_team_size or "null" }}
- mentioned_technologies: {% if previous.mentioned_technologies %}{{ previous.mentioned_technologies | join(", ") }}{% else %}none{% endif %}
- agreed_scope: {{ previous.agreed_scope or "null" }}
{% endif %}
</previous_metadata>

<latest_user_transcript>
{{ transcript }}
</latest_user_transcript>

<latest_assistant_estimation>
summary: {{ result.summary }}
total_duration_weeks: {{ result.total_duration_weeks }}
total_cost_eur: {{ result.total_cost_eur }}
confidence_pct: {{ result.confidence_pct }}
phases:
{% for phase in phases %}
- {{ phase.name }} ({{ phase.duration_weeks }}w, {{ phase.cost_eur }} EUR): {{ phase.summary }}
{% endfor %}
</latest_assistant_estimation>

Extrae el ProjectMetadata de este turno. Devuelve null/lista vacía en los campos que no puedas anclar.
```

**Extractor LLM con salida estructurada.** Elegimos una segunda llamada con `complete_structured(response_model=ProjectMetadata)`: es más robusta que una heurística y reutiliza el mecanismo de la sesión 04. Es tolerante a fallos: si falla, conserva la metadata anterior.

**`app/sessions/metadata_extractor.py`**

```python
import structlog

from app.prompts.loader import render_metadata_extraction_prompt
from app.schemas.estimations import EstimationResult
from app.services.llm_wrapper import LLMWrapper
from app.sessions.models import ProjectMetadata

log = structlog.get_logger()


def update_metadata(
    *,
    previous: ProjectMetadata,
    transcript: str,
    result: EstimationResult,
    llm_wrapper: LLMWrapper,
    model: str,
) -> ProjectMetadata:
    """Extrae metadata del turno y la mezcla con la previa. Si falla, no la toca."""
    system_prompt, user_message = render_metadata_extraction_prompt(
        transcript=transcript, result=result, previous=previous
    )
    try:
        extracted, meta = llm_wrapper.complete_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_model=ProjectMetadata,
            temperature=0.0,
            max_tokens=1000,
            max_retries=2,
            model_override=model,
        )
    except Exception as exc:  # noqa: BLE001 — perder un refresco no debe romper el turno
        log.warning("metadata_extraction_failed", error_type=type(exc).__name__)
        return previous
    return previous.merge_with(extracted)
```

> `model_override` es una extensión pequeña de `complete_structured` para poder usar un modelo barato (`METADATA_EXTRACTOR_MODEL`) distinto del principal. Si no quieres tocarlo, pasa `model` al configurar y usa el principal.

### Paso 5 — Ventana deslizante y adaptación del wrapper

**Config.** En `app/config.py`:

```python
# Sesión 5 — memoria conversacional y adjuntos.
max_conversation_turns: int = 6
max_attachment_chars: int = 60_000
# Vacío => usa el modelo principal del servicio.
metadata_extractor_model: str = ""
```

`ConversationHistory._trim()` ya garantiza la invariante: nunca más de `max_turns * 2` mensajes. `to_messages_list(system_prompt)` devuelve `[system, …ventana]`, con el system regenerado.

**Wrapper.** Añade una variante estructurada que acepte el array `messages`, porque la estimación conversacional envía `system + historial + user`. Refactoriza `complete_structured` para que delegue en ella:

```python
def complete_structured_messages(
    self,
    *,
    messages: list[dict[str, str]],
    response_model: type[StructuredModel],
    temperature: float | None,
    max_tokens: int,
    max_retries: int = 2,
    model_override: str | None = None,
) -> tuple[StructuredModel, dict[str, Any]]:
    """Igual que `complete_structured` pero con un array de mensajes completo."""
    ...  # misma lógica de candidatos/fallback y reintento ante ValidationError


def complete_structured(
    self,
    *,
    system_prompt,
    user_message,
    response_model,
    temperature,
    max_tokens,
    max_retries=2,
    model_override=None,
):
    return self.complete_structured_messages(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_model=response_model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        model_override=model_override,
    )
```

**Pipeline conversacional** en `app/services/llm_service.py`:

```python
@dataclass
class SessionEstimation:
    result: EstimationResult
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    fallback_used: bool = False


def generate_session_estimation(
    *,
    session: Session,
    transcript: str,
    project_type: ProjectType,
    detail_level: DetailLevel,
    output_format: OutputFormat,
    version: str = DEFAULT_ESTIMATION_PROMPT_VERSION,
) -> SessionEstimation:
    """Pipeline multi-turno: sin caché, con memoria y actualización de metadata."""
    _require_configuration()
    if settings.guardrails_enabled:
        check_input(
            transcript, openai_client=get_openai_client() if settings.has_moderation else None
        )

    request = EstimationRequest(
        description=transcript,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
    )
    system_prompt, user_message = render_estimation_prompt(
        request, version, metadata=session.metadata
    )
    messages = session.history.to_messages_list(system_prompt)
    messages.append({"role": "user", "content": user_message})

    wrapper = get_llm_wrapper()
    result, meta = wrapper.complete_structured_messages(
        messages=messages,
        response_model=EstimationResult,
        temperature=settings.temperature,
        max_tokens=settings.llm_max_tokens,
        max_retries=settings.structured_max_retries,
    )
    if settings.guardrails_enabled:
        result = enforce_scope_response(result)

    session.history.append(user=user_message, assistant=result.model_dump_json())
    session.metadata = update_metadata(
        previous=session.metadata,
        transcript=transcript,
        result=result,
        llm_wrapper=wrapper,
        model=settings.metadata_extractor_model or settings.llm_model,
    )
    return SessionEstimation(
        result=result,
        model=str(meta.get("model", "")),
        provider=str(meta.get("provider", "")),
        input_tokens=meta.get("input_tokens"),
        output_tokens=meta.get("output_tokens"),
        cost_usd=meta.get("cost_usd"),
        fallback_used=bool(meta.get("fallback_used", False)),
    )
```

**Contrato de respuesta** en `app/schemas/estimations.py`:

```python
class SessionEstimateResponse(BaseModel):
    session_id: str
    result: EstimationResult
    prompt_version: str
    metadata: ProjectMetadata
    history_messages: int
    model: str = ""
    provider: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    fallback_used: bool = False
```

### Paso 6 — Adaptar el cliente

**`frontend/client.py`**: tres funciones nuevas. La de estimación va en `multipart/form-data`.

```python
def create_session(*, client: httpx.Client | None = None) -> str: ...


def get_session(session_id: str, *, client: httpx.Client | None = None) -> SessionInfo: ...


def estimate_in_session(
    session_id: str,
    *,
    transcript: str,
    project_type: str,
    detail_level: str,
    output_format: str,
    attachments: list[tuple[str, bytes, str]] | None = None,
    prompt_version: str | None = None,
    client: httpx.Client | None = None,
) -> SessionEstimateResponse:
    data = {
        "transcript": transcript,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }
    files = [("attachments", (name, content, mime)) for name, content, mime in (attachments or [])]
    with _acquire_client(client) as http:
        response = http.post(
            _url(f"/api/v1/sessions/{session_id}/estimate"),
            data=data,
            files=files or None,
            params=_version_params(prompt_version),
        )
        _raise_for_status(response)
        return _parse_session_response(response.json())
```

**`frontend/models.py`**: `ProjectMetadata`, `SessionInfo`, `SessionEstimateResponse` (reutilizando `EstimationResult`/`Phase`).

**`frontend/streamlit_app.py`**: pasa a UI conversacional.

- Al cargar la página, si no hay `session_id` en `st.session_state`, llama a `create_session()`.
- Un formulario por turno: `st.text_area` para la transcripción + `st.file_uploader(accept_multiple_files=True, type=["pdf", "docx"])` + los selectores de enums + botón "Añadir turno".
- Los ficheros se leen a bytes y se envían como `attachments`.
- Se acumula el historial de turnos en `st.session_state` y se renderiza en orden.
- El panel lateral muestra la `project_metadata` actual (nombre, equipo, tecnologías, alcance) —el objetivo didáctico es ver la separación entre memoria e historial— y las métricas de la última llamada.
- Botón **"Nueva conversación"** que vuelve a llamar a `create_session()`, resetea el estado y limpia el historial local.

### Paso 7 — Tests mínimos

Añade `asyncio_mode = "auto"` y usa `httpx.AsyncClient` con `ASGITransport` para los tests de integración. El wrapper LLM se sustituye con un doble que implementa `complete_structured_messages` (estimación) y `complete_structured` (extractor).

```python
import httpx
import pytest

from app.main import app


@pytest.fixture
async def async_client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
```

- **`tests/test_sessions_models.py`** (unit): la ventana descarta pares antiguos y conserva los recientes; `to_messages_list` pone el system delante; `merge_with` sobrescribe escalares y une tecnologías sin distinguir mayúsculas; `SessionStore.get_or_404` lanza para un id desconocido.
- **`tests/test_attachments_extractor.py`** (unit): extracción DOCX real (se genera con `python-docx`), truncado a `max_chars`, extensión no soportada → `UnsupportedAttachmentError`, `pypdf` mockeado para PDF, y `enrich_transcript` con/sin adjuntos.
- **`tests/test_sessions_endpoints.py`** (integración async):
  1. **Metadata acumulada:** dos turnos en la misma sesión; el segundo añade una tecnología. `GET /sessions/{id}` refleja la unión y el nombre preservado.
  2. **Adjunto influye:** un DOCX (o PDF mockeado) con un marcador ("Nimbus CRM / React + Postgres") llega al mensaje de usuario del LLM dentro de `--- attachment: … ---`; el test comprueba que el contenido está en `messages`.
  3. **Ventana acotada:** 8 turnos con `max_conversation_turns=3`; el array `messages` enviado en cada estimación nunca supera `1 (system) + 3*2 (ventana) + 1 (turno actual)` y el historial de la sesión queda en `≤ 3*2`.
- **`tests/test_sessions_404.py`** (o dentro del anterior): sesión inexistente → 404; adjunto no soportado → 415.

## Criterios de "hecho"

El ejercicio está completo cuando:

- [ ] `POST /api/v1/sessions` crea una sesión y devuelve `session_id`.
- [ ] `POST /api/v1/sessions/{session_id}/estimate` acepta `multipart/form-data` con transcripción y adjuntos opcionales, y devuelve una estimación que respeta `EstimationResult`.
- [ ] Tras varios turnos en la misma sesión, el modelo responde con coherencia respecto al proyecto (no olvida el nombre entre turnos).
- [ ] `project_metadata` se actualiza visiblemente entre turnos.
- [ ] El historial respeta el límite de la ventana deslizante.
- [ ] El README indica el Camino B y cómo se extrae la metadata.
- [ ] Los tests del Paso 7 pasan en local.

## Checklist de verificación

- [ ] `app/sessions/` con `models.py`, `store.py`, `metadata_extractor.py` y docstrings de volatilidad.
- [ ] El system prompt lleva `<project_metadata>` (vacío en el primer turno); no se almacena en el historial.
- [ ] `ConversationHistory.to_messages_list(system_prompt)` devuelve `[system, …ventana]`.
- [ ] `MAX_CONVERSATION_TURNS` es configurable y el recorte respeta los pares.
- [ ] `extract_text` soporta PDF y DOCX, trunca a `max_attachment_chars` y distingue 415 de 422.
- [ ] `enrich_transcript` delimita cada adjunto con `--- attachment: … ---`/`--- end attachment ---`.
- [ ] `complete_structured_messages` existe y `complete_structured` delega en él.
- [ ] El endpoint de sesión no cachea; el `/estimate` de formulario sigue igual.
- [ ] La metadata se actualiza con extractor LLM y, si falla, se conserva la anterior.
- [ ] El cliente crea sesión, permite adjuntos, muestra la metadata y tiene "Nueva conversación".
- [ ] `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` y `uv run mypy app frontend` pasan.

## Documentación de referencia

- FastAPI — formularios y ficheros (`Form`, `File`, `UploadFile`, multipart): https://fastapi.tiangolo.com/tutorial/request-files/
- pypdf — extracción de texto: https://pypdf.readthedocs.io/en/stable/user/extract-text.html
- python-docx — lectura de párrafos: https://python-docx.readthedocs.io/en/latest/
- httpx — cliente asíncrono y `ASGITransport`: https://www.python-httpx.org/async/
- pytest-asyncio — modo automático: https://pytest-asyncio.readthedocs.io/

## Nota

**Historial vs memoria.** El *historial* es el array `messages` que viaja a la API y está acotado por la ventana deslizante; se olvida. La *memoria* son los hechos del proyecto (`project_metadata`) que sobreviven a los turnos y se inyectan en el system prompt. Enviar el historial bruto completo crece sin límite y encarece cada turno; destilar los hechos a metadata mantiene el coste acotado y da al modelo un contexto estable. La ventana deslizante es la estrategia por defecto razonable para arrancar; sus problemas concretos (pierde información antigua relevante, no distingue lo importante de lo accesorio) son los que empujan al resumen acumulativo con anclas y al tier dinámico del directo.

**Por qué Camino B.** Extraer el texto localmente desacopla el servicio del proveedor (funciona igual con OpenAI y Anthropic), permite testear sin red y deja el texto preparado para el chunking de RAG. A cambio, renunciamos a la comprensión multimodal de diagramas y esquemas que daría subir el PDF directo al modelo.

**Por qué extractor LLM.** Una heurística por regex es barata pero frágil (nombre del proyecto, alcance). El extractor con salida estructurada reutiliza el mecanismo validado de la sesión 04, es más robusto y, al ejecutarse con un modelo barato y tolerar fallos, el coste extra por turno es asumible.

**Fuera de alcance.** Resumen acumulativo con anclas, tier dinámico, persistencia de la memoria entre reinicios, búsqueda web, function calling y el patrón Actor-Critic-Boss se construyen en el directo o en módulos posteriores.
