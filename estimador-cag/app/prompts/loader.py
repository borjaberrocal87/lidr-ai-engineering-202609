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

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimations import EstimationRequest

_BASE_DIR = Path(__file__).resolve().parent

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
    system = _env.get_template(f"estimation/{version}/system.j2").render(**context)
    user = _env.get_template(f"estimation/{version}/user.j2").render(**context)
    return system, user
