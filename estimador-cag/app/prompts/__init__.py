"""Prompts versionados en plantillas Jinja2.

La estructura en disco es `app/prompts/<caso_de_uso>/<version>/<rol>.j2`. La API
pública es `render_estimation_prompt`, que devuelve el par `(system, user)`.
"""

from app.prompts.loader import DEFAULT_ESTIMATION_PROMPT_VERSION, render_estimation_prompt

__all__ = ["DEFAULT_ESTIMATION_PROMPT_VERSION", "render_estimation_prompt"]
