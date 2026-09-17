"""Servicio de llamada al LLM con arquitectura CAG.

El contexto estático (ejemplos de estimaciones previas) se inyecta en el
system prompt (parámetro `instructions` en la Responses API) en cada llamada.
La transcripción de la nueva reunión viaja como `input`.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass

import structlog

from app.config import settings
from app.context.examples import ESTIMATION_EXAMPLES

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


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
    log = _call_logger(transcription)
    log.info("llm.call.start", temperature=settings.temperature)
    started_at = time.perf_counter()

    try:
        result = _dispatch_generate(transcription)
    except Exception:
        log.exception("llm.call.error", latency_ms=_elapsed_ms(started_at))
        raise

    log.info(
        "llm.call.end",
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=_elapsed_ms(started_at),
    )
    return result


def _dispatch_generate(transcription: str) -> EstimationResult:
    if settings.llm_provider == "openai":
        return _estimate_with_openai(transcription)
    if settings.llm_provider == "anthropic":
        return _estimate_with_anthropic(transcription)
    if settings.llm_provider == "custom":
        return _estimate_with_openai_compatible(transcription)
    raise LLMConfigurationError(
        f"Proveedor LLM no soportado: {settings.llm_provider!r}. "
        "Usa 'openai', 'anthropic' o 'custom' en LLM_PROVIDER."
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
    log = _call_logger(transcription)
    log.info("llm.stream.start", temperature=settings.temperature)
    started_at = time.perf_counter()

    active_metrics = metrics if metrics is not None else StreamMetrics(model="", provider="")
    try:
        inner = _dispatch_stream(transcription, active_metrics)
    except Exception:
        log.exception("llm.stream.error", latency_ms=_elapsed_ms(started_at))
        raise

    return _logged_stream(inner, log, active_metrics, started_at)


def _dispatch_stream(
    transcription: str,
    metrics: StreamMetrics | None,
) -> Iterator[str]:
    if settings.llm_provider == "openai":
        _require_openai_key()
        return _stream_with_openai(transcription, metrics)
    if settings.llm_provider == "anthropic":
        _require_anthropic_key()
        return _stream_with_anthropic(transcription, metrics)
    if settings.llm_provider == "custom":
        _require_custom_base_url()
        _require_custom_key()
        return _stream_with_openai_compatible(transcription, metrics)
    raise LLMConfigurationError(
        f"Proveedor LLM no soportado: {settings.llm_provider!r}. "
        "Usa 'openai', 'anthropic' o 'custom' en LLM_PROVIDER."
    )


def _logged_stream(
    inner: Iterator[str],
    log: structlog.stdlib.BoundLogger,
    metrics: StreamMetrics | None,
    started_at: float,
) -> Iterator[str]:
    """Envuelve el generador del proveedor para registrar el fin del streaming."""
    try:
        yield from inner
    except Exception:
        log.exception("llm.stream.error", latency_ms=_elapsed_ms(started_at))
        raise
    except BaseException:
        log.warning("llm.stream.aborted", latency_ms=_elapsed_ms(started_at))
        raise

    log.info(
        "llm.stream.end",
        input_tokens=getattr(metrics, "input_tokens", None),
        output_tokens=getattr(metrics, "output_tokens", None),
        latency_ms=_elapsed_ms(started_at),
    )


def _call_logger(transcription: str) -> structlog.stdlib.BoundLogger:
    """Logger con el contexto común de cada llamada al LLM (sin datos sensibles)."""
    return logger.bind(
        provider=settings.llm_provider,
        model=settings.llm_model,
        transcription_chars=len(transcription),
    )


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)


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
    # La Chat Completions API (proveedor custom) usa otros nombres.
    if metrics.input_tokens is None:
        metrics.input_tokens = getattr(usage, "prompt_tokens", None)
    if metrics.output_tokens is None:
        metrics.output_tokens = getattr(usage, "completion_tokens", None)


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


def _stream_with_openai_compatible(
    transcription: str,
    metrics: StreamMetrics | None,
) -> Iterator[str]:
    from openai import OpenAI

    client = OpenAI(
        api_key=_require_custom_key(),
        base_url=_require_custom_base_url(),
    )
    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": transcription},
        ],
        temperature=settings.temperature,
        stream=True,
        stream_options={"include_usage": True},
    )

    usage = None
    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

    _record_metrics(
        metrics,
        model=settings.llm_model,
        provider="custom",
        usage=usage,
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


def _estimate_with_openai_compatible(transcription: str) -> EstimationResult:
    from openai import OpenAI

    client = OpenAI(
        api_key=_require_custom_key(),
        base_url=_require_custom_base_url(),
    )
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": transcription},
        ],
        temperature=settings.temperature,
    )

    message = response.choices[0].message
    usage = response.usage
    return EstimationResult(
        estimation=message.content or "",
        model=settings.llm_model,
        provider="custom",
        temperature=settings.temperature,
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
    )


def _require_custom_base_url() -> str:
    if not settings.custom_llm_base_url:
        raise LLMConfigurationError(
            "CUSTOM_LLM_BASE_URL no está configurada. Añádela al archivo .env."
        )
    return settings.custom_llm_base_url


def _require_custom_key() -> str:
    if not settings.custom_llm_api_key:
        raise LLMConfigurationError(
            "CUSTOM_LLM_API_KEY no está configurada. Añádela al archivo .env."
        )
    return settings.custom_llm_api_key
