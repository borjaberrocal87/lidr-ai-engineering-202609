"""Servicio de llamada al LLM con arquitectura CAG.

El contexto estático (ejemplos de estimaciones previas) se inyecta en el
system prompt (parámetro `instructions` en la Responses API) en cada llamada.
La transcripción de la nueva reunión viaja como `input`.
"""

from collections.abc import Iterator
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
    temperature: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class StreamMetrics:
    """Métricas de una generación en streaming.

    Se rellena durante el consumo del generador: tras agotarlo, el llamante
    puede leer los tokens y el modelo utilizados.
    """

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


def stream_estimation(
    transcription: str,
    metrics: StreamMetrics | None = None,
) -> Iterator[str]:
    """Genera una estimación en streaming, cediendo el texto token a token.

    Valida la configuración del proveedor de forma anticipada (al llamar a la
    función), de modo que los errores de configuración se propagan antes de
    empezar a consumir el generador. Si se pasa `metrics`, se rellena con el
    modelo y los tokens utilizados una vez agotado el generador.
    """
    if settings.llm_provider == "openai":
        _require_openai_key()
        return _stream_with_openai(transcription, metrics)
    if settings.llm_provider == "anthropic":
        _require_anthropic_key()
        return _stream_with_anthropic(transcription, metrics)
    raise LLMConfigurationError(
        f"Proveedor LLM no soportado: {settings.llm_provider!r}. "
        "Usa 'openai' o 'anthropic' en LLM_PROVIDER."
    )


def _record_metrics(
    metrics: StreamMetrics | None,
    *,
    model: str,
    provider: str,
    usage: object | None,
) -> None:
    if metrics is None:
        return
    metrics.model = model
    metrics.provider = provider
    metrics.input_tokens = getattr(usage, "input_tokens", None)
    metrics.output_tokens = getattr(usage, "output_tokens", None)


def _stream_with_openai(
    transcription: str,
    metrics: StreamMetrics | None,
) -> Iterator[str]:
    from openai import OpenAI

    client = OpenAI(api_key=_require_openai_key())
    with client.responses.stream(
        model=settings.llm_model,
        instructions=build_system_prompt(),
        input=transcription,
        temperature=settings.temperature,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta
        response = stream.get_final_response()

    _record_metrics(
        metrics,
        model=settings.llm_model,
        provider="openai",
        usage=getattr(response, "usage", None),
    )


def _stream_with_anthropic(
    transcription: str,
    metrics: StreamMetrics | None,
) -> Iterator[str]:
    from anthropic import Anthropic

    client = Anthropic(api_key=_require_anthropic_key())
    with client.messages.stream(
        model=settings.llm_model,
        max_tokens=2048,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": transcription}],
    ) as stream:
        yield from stream.__stream_text__()
        final = stream.get_final_message()

    _record_metrics(
        metrics,
        model=settings.llm_model,
        provider="anthropic",
        usage=getattr(final, "usage", None),
    )


def _require_openai_key() -> str:
    if not settings.open_ai_key:
        raise LLMConfigurationError("OPEN_AI_KEY no está configurada. Añádela al archivo .env.")
    return settings.open_ai_key


def _require_anthropic_key() -> str:
    if not settings.anthropic_api_key:
        raise LLMConfigurationError(
            "ANTHROPIC_API_KEY no está configurada. Añádela al archivo .env."
        )
    return settings.anthropic_api_key


def _estimate_with_openai(transcription: str) -> EstimationResult:
    from openai import OpenAI

    client = OpenAI(api_key=_require_openai_key())
    response = client.responses.create(
        model=settings.llm_model,
        instructions=build_system_prompt(),
        input=transcription,
        temperature=settings.temperature,
    )

    usage = response.usage
    return EstimationResult(
        estimation=response.output_text or "",
        model=settings.llm_model,
        provider="openai",
        temperature=settings.temperature,
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )


def _estimate_with_anthropic(transcription: str) -> EstimationResult:
    from anthropic import Anthropic

    client = Anthropic(api_key=_require_anthropic_key())
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=2048,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": transcription}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    usage = response.usage
    return EstimationResult(
        estimation=text,
        model=settings.llm_model,
        provider="anthropic",
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
    )
