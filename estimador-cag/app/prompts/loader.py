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

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimations import EstimationRequest

_BASE_DIR = Path(__file__).resolve().parent

ESTIMATION_USE_CASE = "estimation"

DEFAULT_ESTIMATION_PROMPT_VERSION = "v1"

_env = Environment(
    loader=FileSystemLoader(_BASE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,  # noqa: S701 — son prompts de texto, no HTML
    keep_trailing_newline=True,
)


def render_estimation_prompt(
    request: EstimationRequest,
    version: str = DEFAULT_ESTIMATION_PROMPT_VERSION,
) -> tuple[str, str]:
    """Renderiza el par ``(system_prompt, user_prompt)`` del caso de estimación.

    Devuelve dos strings listos para enviar al modelo como mensajes separados
    ``role: "system"`` y ``role: "user"``. Cambiar de versión no obliga a tocar
    el llamante: basta con pasar ``version="v2"``.
    """
    context = {
        "description": request.description,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
    }
    system = _env.get_template(f"{ESTIMATION_USE_CASE}/{version}/system.j2").render(**context)
    user = _env.get_template(f"{ESTIMATION_USE_CASE}/{version}/user.j2").render(**context)
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
