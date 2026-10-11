"""Cargador de plantillas de prompt versionadas.

El layout en disco es ``app/prompts/<caso_de_uso>/<version>/<rol>.j2``. El
versionado existe desde el primer día: cambiar de prompt es cambiar el string
``version`` en el llamante, no refactorizar código.

Las plantillas se renderizan con ``StrictUndefined`` para que un typo entre el
contexto y la plantilla reviente en el render en lugar de interpolarse vacío, y
con ``trim_blocks``/``lstrip_blocks`` para que las etiquetas de control
(``{% if %}``, ``{% include %}``) no dejen saltos ni espacios en el prompt.
"""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimations import (
    DetailLevel,
    EstimationRequest,
    EstimationResult,
    OutputFormat,
    ProjectType,
)
from app.sessions.models import ProjectMetadata

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_BASE_DIR = Path(__file__).resolve().parent

ESTIMATION_USE_CASE = "estimation"
METADATA_EXTRACTION_USE_CASE = "metadata_extraction"
CRITIC_USE_CASE = "critic"

DEFAULT_ESTIMATION_PROMPT_VERSION = "v1"
# Versión por defecto del prompt conversacional (sesión 5). Se elevará a "v3"
# cuando el prompt con bloque <audience> entre en juego (Fase C del directo).
DEFAULT_CONVERSATIONAL_PROMPT_VERSION = "v1"

_env = Environment(
    loader=FileSystemLoader(_BASE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,  # noqa: S701 — son prompts de texto, no HTML
    keep_trailing_newline=True,
)


def _render_estimation_templates(
    version: str,
    context: dict[str, object],
    *,
    reference_projects_count: int,
) -> tuple[str, str]:
    """Renderiza system+user de la versión dada y emite el evento de trazabilidad."""
    system = _env.get_template(f"{ESTIMATION_USE_CASE}/{version}/system.j2").render(**context)
    user = _env.get_template(f"{ESTIMATION_USE_CASE}/{version}/user.j2").render(**context)
    logger.info(
        "prompt.rendered",
        use_case=ESTIMATION_USE_CASE,
        version=version,
        system_chars=len(system),
        user_chars=len(user),
        system_hash=_content_hash(system),
        user_hash=_content_hash(user),
        prompt_fingerprint=prompt_fingerprint(version),
        reference_projects=reference_projects_count,
    )
    return system, user


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = DEFAULT_ESTIMATION_PROMPT_VERSION,
    metadata: ProjectMetadata | None = None,
) -> tuple[str, str]:
    """Renderiza el par ``(system_prompt, user_prompt)`` del caso de estimación.

    Devuelve dos strings listos para enviar al modelo como mensajes separados
    ``role: "system"`` y ``role: "user"``. Cambiar de versión no obliga a tocar
    el llamante: basta con pasar ``version="v2"``. `metadata` alimenta el bloque
    `<project_metadata>` del system prompt; si es `None` o está vacía, el bloque
    no se renderiza.
    """
    context: dict[str, object] = {
        "description": request.description,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "reference_projects": [
            project.model_dump() for project in (request.reference_projects or [])
        ],
        "metadata": metadata,
        "metadata_is_empty": metadata is None or metadata.is_empty(),
    }
    return _render_estimation_templates(
        version, context, reference_projects_count=len(request.reference_projects or [])
    )


def render_conversational_prompt(
    *,
    description: str,
    project_type: ProjectType,
    detail_level: DetailLevel,
    output_format: OutputFormat,
    metadata: ProjectMetadata,
    version: str = DEFAULT_ESTIMATION_PROMPT_VERSION,
    tier: object | None = None,
    critic_feedback: object | None = None,
) -> tuple[str, str]:
    """Renderiza el prompt de un turno conversacional (sesión 5).

    A diferencia de `render_estimation_prompt`, recibe los campos sueltos: el
    texto enriquecido con adjuntos puede superar los límites de descripción del
    formulario, así que evitamos construir un `EstimationRequest` que fallaría
    en la validación. La `metadata` siempre se inyecta (vacía en el primer
    turno), lo que activa además las reglas conversacionales del system prompt.

    `tier` y `critic_feedback` solo los consume la versión ``v3`` (bloque
    ``<audience>`` y feedback del Critic en el bucle ACB); las versiones
    anteriores los ignoran.
    """
    context: dict[str, object] = {
        "description": description,
        "project_type": project_type.value,
        "detail_level": detail_level.value,
        "output_format": output_format.value,
        "reference_projects": [],
        "metadata": metadata,
        "metadata_is_empty": metadata.is_empty(),
        "tier": _enum_value(tier),
        "critic_feedback": critic_feedback,
    }
    return _render_estimation_templates(version, context, reference_projects_count=0)


def render_metadata_extraction_prompt(
    *,
    transcript: str,
    result: EstimationResult,
    previous: ProjectMetadata,
    version: str = "v1",
) -> tuple[str, str]:
    """Renderiza los prompts del extractor de `ProjectMetadata` (sesión 5).

    Es una segunda llamada LLM por turno: lee la última transcripción, la
    estimación producida y la metadata acumulada, y devuelve un
    `ProjectMetadata` parcial (validado por el wrapper estructurado).
    """
    context = {
        "transcript": transcript,
        "result": result,
        "phases": result.phases,
        "previous": previous,
        "previous_is_empty": previous.is_empty(),
    }
    system = _env.get_template(f"{METADATA_EXTRACTION_USE_CASE}/{version}/system.j2").render(
        **context
    )
    user = _env.get_template(f"{METADATA_EXTRACTION_USE_CASE}/{version}/user.j2").render(**context)
    return system, user


def _content_hash(text: str) -> str:
    """Hash corto del contenido, para trazar el render sin registrar su texto."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _enum_value(value: object | None) -> object | None:
    """Devuelve el ``.value`` de un Enum (str) o el propio valor si no lo es."""
    return getattr(value, "value", value)


def render_critic_prompt(
    *,
    transcript: str,
    metadata: ProjectMetadata,
    tier: object,
    result: EstimationResult,
    version: str = "v1",
) -> tuple[str, str]:
    """Renderiza los prompts del Critic (patrón Actor-Critic-Boss, sesión 5).

    El Critic audita una estimación ya producida: recibe la transcripción, la
    metadata acumulada, el tier resuelto y el resultado bajo revisión, y devuelve
    un `CriticFeedback` estructurado.
    """
    context: dict[str, object] = {
        "transcript": transcript,
        "metadata": metadata,
        "tier": _enum_value(tier),
        "result": result,
        "phases": result.phases,
    }
    system = _env.get_template(f"{CRITIC_USE_CASE}/{version}/system.j2").render(**context)
    user = _env.get_template(f"{CRITIC_USE_CASE}/{version}/user.j2").render(**context)
    logger.info(
        "prompt.rendered",
        use_case=CRITIC_USE_CASE,
        version=version,
        system_chars=len(system),
        user_chars=len(user),
        system_hash=_content_hash(system),
        user_hash=_content_hash(user),
    )
    return system, user


def available_estimation_versions() -> list[str]:
    """Versiones de prompt de estimación presentes en disco, ordenadas.

    Una versión cuenta si tiene `system.j2` y `user.j2`. Lo usa el router para
    validar el query param `?prompt_version=` y `/context` para exponerlas.
    """
    directory = _BASE_DIR / ESTIMATION_USE_CASE
    if not directory.is_dir():
        return []
    versions = [
        child.name
        for child in directory.iterdir()
        if child.is_dir() and (child / "system.j2").is_file() and (child / "user.j2").is_file()
    ]
    return sorted(versions)


@cache
def prompt_fingerprint(version: str = DEFAULT_ESTIMATION_PROMPT_VERSION) -> str:
    """Huella determinista de las **fuentes** de una versión del prompt.

    Se calcula sobre los ficheros ``.j2`` de la versión (no sobre el render, que
    varía por petición). Sirve como identidad del artefacto de prompt para la
    clave de caché: cualquier edición de un template cambia la huella e invalida
    la caché, aunque no se suba el número de versión.
    """
    directory = _BASE_DIR / ESTIMATION_USE_CASE / version
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.j2")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]
