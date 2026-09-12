"""Servicio de llamada al LLM con arquitectura CAG.

El contexto estático (ejemplos de estimaciones previas) se inyecta en el
system prompt en cada llamada. La transcripción de la nueva reunión viaja
como mensaje de usuario.
"""

from dataclasses import dataclass

from app.config import settings
from app.context.examples import ESTIMATION_EXAMPLES


class LLMConfigurationError(RuntimeError):
    """Error de configuración del proveedor LLM (API key ausente, etc.)."""


@dataclass
class EstimationResult:
    estimation: str
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None


SYSTEM_PROMPT = """Eres un estimador de software experto. Tu tarea es analizar la \
transcripción de una reunión con un cliente y producir una estimación de software \
profesional, clara y accionable.

Reglas:
- Usa los ejemplos de estimaciones previas como referencia de formato, nivel de \
detalle y criterio de esfuerzo.
- Desglosa el trabajo en tareas concretas con horas estimadas por tarea.
- Incluye el total de horas, el equipo recomendado, la duración estimada y un \
rango de coste cuando sea posible.
- Si la transcripción es ambigua, explicita los supuestos que asumes.
- Responde en español y en formato Markdown."""


def _format_examples() -> str:
    blocks = []
    for index, example in enumerate(ESTIMATION_EXAMPLES, start=1):
        blocks.append(
            f"### Ejemplo {index}\n"
            f"**Resumen de la reunión:**\n{example['meeting_summary']}\n\n"
            f"**Estimación generada:**\n{example['estimation']}"
        )
    return "\n\n".join(blocks)


def build_system_prompt() -> str:
    """Construye el system prompt inyectando el contexto estático (CAG)."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "## Ejemplos de estimaciones previas (contexto de referencia)\n\n"
        f"{_format_examples()}"
    )


def generate_estimation(transcription: str) -> EstimationResult:
    """Genera una estimación a partir de una transcripción usando el proveedor configurado."""
    if settings.llm_provider == "openai":
        return _estimate_with_openai(transcription)
    if settings.llm_provider == "anthropic":
        return _estimate_with_anthropic(transcription)
    raise LLMConfigurationError(
        f"Proveedor LLM no soportado: {settings.llm_provider!r}. "
        "Usa 'openai' o 'anthropic' en LLM_PROVIDER."
    )


def _estimate_with_openai(transcription: str) -> EstimationResult:
    if not settings.open_ai_key:
        raise LLMConfigurationError(
            "OPEN_AI_KEY no está configurada. Añádela al archivo .env."
        )

    from openai import OpenAI

    client = OpenAI(api_key=settings.open_ai_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": transcription},
        ],
        temperature=0.2,
    )

    usage = response.usage
    return EstimationResult(
        estimation=response.choices[0].message.content or "",
        model=settings.llm_model,
        provider="openai",
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
    )


def _estimate_with_anthropic(transcription: str) -> EstimationResult:
    if not settings.anthropic_api_key:
        raise LLMConfigurationError(
            "ANTHROPIC_API_KEY no está configurada. Añádela al archivo .env."
        )

    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=2048,
        temperature=0.2,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": transcription}],
    )

    text = "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    )
    usage = response.usage
    return EstimationResult(
        estimation=text,
        model=settings.llm_model,
        provider="anthropic",
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )
