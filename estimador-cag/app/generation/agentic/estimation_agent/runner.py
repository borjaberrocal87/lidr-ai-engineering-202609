"""El bucle manual del agente de estimación (Sesión 12).

Un desarrollador senior reconoce todo lo que hay aquí: un bucle que llama a un
LLM que **decide**, ejecuta **tools** y para cuando ha terminado. Sin framework.

Excepción deliberada a la convención del repo: el resto de llamadas al LLM pasan
por ``LLMWrapper`` (LiteLLM + Instructor). Este módulo habla directamente con la
Responses API de OpenAI porque el ejercicio consiste en conducir a mano el
razona -> actúa -> observa para poder capturar cada paso. No lo "arregles" para
usar ``LLMWrapper``: se perdería la traza.

Mecánica (encadenado con estado en servidor):

1. ``responses.create`` con la transcripción y los schemas de tools. El modelo
   emite items ``reasoning`` y ``function_call`` y se detiene esperándonos.
2. Se recorren los ``function_call`` de ``response.output``, se ejecuta la función
   correspondiente y se devuelve un ``function_call_output`` por cada ``call_id``.
3. Se vuelve a llamar con ``previous_response_id`` y SOLO los outputs nuevos: el
   servidor conserva el razonamiento y las calls previas con su orden.
4. Se repite hasta que una vuelta no traiga ``function_call`` (parada natural) o
   se alcance ``max_iterations`` (salvaguarda).
5. Un último ``responses.parse`` convierte el contexto acumulado en un
   ``AgentEstimateReport`` validado.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from app.generation.agentic.estimation_agent.schemas import (
    AgentEstimateReport,
    AgentRunResult,
    AgentStepRecord,
    AgentTrace,
)
from app.generation.agentic.estimation_agent.tools import (
    TOOL_SCHEMAS,
    BudgetSearchBackend,
    dispatch_tool,
)

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """\
You are an estimation agent for a software consultancy. You receive the raw \
transcript of a discovery meeting and must produce a grounded effort estimate in \
engineer-hours.

Method - follow it step by step:
1. Read the transcript and DECOMPOSE the project into its distinct components. \
Real projects usually mix pieces of different nature: a business backend, an ERP \
or third-party integration, a mobile app, an analytics dashboard. List every \
component the client wants estimated; do not merge unrelated pieces into one.
2. For EACH component, call `search_budgets` with a focused, component-specific \
query to retrieve how much analogous work cost historically, in engineer-hours. \
Make one search per component. Do not try to cover the whole project in a single \
broad query.
3. Once you have reference hours for every component, call `calculate_estimate` \
with all of them and their reference amounts to get the partial and total \
breakdown.
4. Call `validate_estimate` as the LAST tool step and fix whatever it flags: a \
component with no historical reference must be searched again, and a total that \
does not match its components must be recalculated.
5. When the estimate is clean, stop calling tools. You will then be asked to \
return the final structured report.

You have exactly these tools: `search_budgets`, `calculate_estimate`, \
`validate_estimate`. Ground every number in what `search_budgets` returns. When \
the transcript does not specify something you need, record it as an assumption \
instead of silently inventing it.
"""

FINAL_INSTRUCTION = (
    "Return the final structured report now, consolidating the components you "
    "costed. Set total_hours to the sum of the component hours, list the "
    "assumptions you had to make, and choose a confidence level that reflects how "
    "well the historical budgets matched the requested work. For each component, "
    "include the database ids of the historical chunks that grounded it."
)


def _summarize_reasoning(output: list[Any]) -> str | None:
    """Concatena el reasoning summary del turno, si la API emitió alguno.

    La Responses API solo devuelve summaries si la llamada pasa
    ``reasoning={"summary": "auto"}``; incluso así pueden llegar vacíos con
    esfuerzos baratos.
    """
    parts: list[str] = []
    for item in output:
        if getattr(item, "type", None) != "reasoning":
            continue
        for summary in getattr(item, "summary", None) or []:
            text = getattr(summary, "text", None)
            if text:
                parts.append(str(text))
    return " ".join(parts) if parts else None


def _function_calls(output: list[Any]) -> list[Any]:
    return [item for item in output if getattr(item, "type", None) == "function_call"]


def _observation(result: dict[str, Any]) -> str:
    summary = result.get("summary")
    if summary:
        return str(summary)
    if result.get("error"):
        return str(result["error"])
    return json.dumps(result, ensure_ascii=False)[:200]


async def run_estimation_agent(
    transcript: str,
    *,
    client: Any,
    model: str,
    reasoning_effort: str = "medium",
    max_iterations: int = 10,
    backend: BudgetSearchBackend,
) -> AgentRunResult:
    """Conduce el bucle sobre una transcripción y devuelve informe + traza.

    ``client`` es un ``AsyncOpenAI``; ``backend`` decide cómo se resuelve
    ``search_budgets`` (pipeline real o stub offline). Ambos se inyectan para que
    la capa no dependa de infraestructura.
    """
    iteration_budget = max(1, max_iterations)
    trace = AgentTrace(model=model)
    step_number = 0
    stopped_reason: str = "completed"

    log.info(
        "agent.run.start",
        model=model,
        effort=reasoning_effort,
        max_iterations=iteration_budget,
    )
    response = await client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": transcript}],
        tools=TOOL_SCHEMAS,
        reasoning={"effort": reasoning_effort, "summary": "auto"},
        store=True,
    )
    iterations = 1

    while True:
        calls = _function_calls(response.output)
        if not calls:
            break
        if iterations >= iteration_budget:
            stopped_reason = "max_iterations"
            log.warning("agent.run.max_iterations", iterations=iterations)
            break

        # El modelo razona una vez por turno aunque emita varias calls en
        # paralelo. El summary se cuelga del primer paso y los hermanos se marcan
        # como calls del mismo turno, en vez de repetir el bloque entero.
        reasoning = _summarize_reasoning(response.output)
        first_step = step_number + 1
        tool_outputs: list[dict[str, Any]] = []

        for call in calls:
            step_number += 1
            step_reasoning = (
                reasoning
                if step_number == first_step
                else f"(parallel tool call in the same turn as STEP {first_step})"
            )
            tool_name = getattr(call, "name", "unknown")
            try:
                raw_args = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                raw_args = {}
                result: dict[str, Any] = {"error": f"arguments were not valid JSON: {exc}"}
            else:
                try:
                    result = await dispatch_tool(tool_name, raw_args, backend=backend)
                except Exception as exc:
                    log.warning("agent.tool.error", tool=tool_name, error=str(exc)[:200])
                    result = {"error": f"{type(exc).__name__}: {exc}"}

            trace.steps.append(
                AgentStepRecord(
                    step=step_number,
                    reasoning_summary=step_reasoning,
                    tool=tool_name,
                    tool_args=raw_args,
                    observation=_observation(result),
                )
            )
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, ensure_ascii=False, default=str),
                }
            )

        response = await client.responses.create(
            model=model,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=TOOL_SCHEMAS,
            reasoning={"effort": reasoning_effort, "summary": "auto"},
            store=True,
        )
        iterations += 1

    report: AgentEstimateReport | None = None
    if stopped_reason != "max_iterations":
        try:
            parsed = await client.responses.parse(
                model=model,
                previous_response_id=response.id,
                input=[{"role": "user", "content": FINAL_INSTRUCTION}],
                text_format=AgentEstimateReport,
                store=True,
            )
            report = parsed.output_parsed
            iterations += 1
        except Exception as exc:
            log.error("agent.run.final_parse_failed", error=str(exc)[:300])
            stopped_reason = "no_final_report"

    if report is None and stopped_reason == "completed":
        stopped_reason = "no_final_report"

    trace.iterations = iterations
    trace.stopped_reason = stopped_reason  # type: ignore[assignment]
    log.info(
        "agent.run.done",
        iterations=iterations,
        steps=len(trace.steps),
        stopped_reason=stopped_reason,
        total_hours=(report.total_hours if report else None),
    )
    return AgentRunResult(
        report=report,
        trace=trace,
        iterations=iterations,
        stopped_reason=stopped_reason,  # type: ignore[arg-type]
    )
