"""Fase 2 del agente: recuperación de horas para las tareas no groundadas (S12).

El paso determinista por tarea (``task_hours.estimate_all``) corre primero. Solo
las tareas que quedaron sin análogo — o con una fiabilidad demasiado baja — se
entregan a este bucle, que es donde el agente se gana el sueldo: reformula la
búsqueda, reúne análogos y deriva horas con la MISMA matemática de consenso que
el camino determinista.

Mecánica (encadenado con estado en servidor, como el agente autónomo):

1. ``responses.create`` con la lista de tareas marcadas y los schemas de tools.
2. Se ejecutan los ``function_call`` (``search_budgets`` / ``derive_task_hours`` /
   ``validate_estimate``) y se devuelve un ``function_call_output`` por ``call_id``.
3. Se vuelve a llamar con ``previous_response_id`` y solo los outputs nuevos.
4. Se repite hasta que una vuelta no traiga ``function_call`` o se alcance
   ``max_iterations``.

No hay parse final: las horas recuperadas salen de las observaciones de
``derive_task_hours`` (deterministas), nunca de una cifra que el modelo invente.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from app.domain.schemas.agent_trace import AgentStep, AgentTrace
from app.generation.agentic.estimation_agent._loop import (
    function_calls,
    observation,
    summarize_reasoning,
)
from app.generation.agentic.estimation_agent.schemas import (
    AgentTaskDerivation,
    AgentTaskHoursRun,
    AgentTaskRef,
    DeriveTaskHoursArgs,
)
from app.generation.agentic.estimation_agent.tools import (
    RECOVERY_TOOL_SCHEMAS,
    BudgetSearchBackend,
    ConsensusFn,
    dispatch_tool,
)

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """\
You are an estimation agent for a software consultancy. The deterministic \
per-task pass could NOT ground the tasks in the list below: it either found no \
historical analog or matched one too weakly. Your job is to recover believable \
hours for each flagged task, one at a time.

Method - follow it task by task:
1. For each flagged task, reformulate a focused query (mention the module, the \
task and the key technology or domain) and call `search_budgets` to retrieve \
historical analogs with recorded engineer-hours and their cosine distance.
2. Once a task has relevant analogs, call `derive_task_hours` with that task and \
the analogs (each with its recorded hours and distance). This tool applies the \
SAME distance-weighted consensus as the deterministic pipeline: do NOT invent \
hours yourself and do NOT call `calculate_estimate`.
3. If the analogs are weak or empty, reformulate and search again before deriving. \
Never pass neighbors you do not trust.
4. When every flagged task has been derived (or you conclude it cannot be \
grounded), stop calling tools.

You have exactly these tools: `search_budgets`, `derive_task_hours`, \
`validate_estimate`. Ground every number in what `search_budgets` returns.
"""


def _user_message(tasks: list[AgentTaskRef]) -> str:
    lines = ["Flagged tasks (module | task | description):"]
    for task in tasks:
        description = task.description or "(no description)"
        lines.append(f"- {task.module} | {task.task} | {description}")
    lines.append("")
    lines.append("Recover the hours for each of these tasks now.")
    return "\n".join(lines)


def _parse_derivation(result: dict[str, Any], reasoning: str | None) -> AgentTaskDerivation | None:
    """Convierte la observación de ``derive_task_hours`` en una derivación tipada."""
    if "estimated_hours" not in result or "module" not in result or "task" not in result:
        return None
    try:
        return AgentTaskDerivation(
            module=str(result["module"]),
            task=str(result["task"]),
            estimated_hours=int(result["estimated_hours"]),
            reliability=float(result.get("reliability") or 0.0),
            dispersion=float(result.get("dispersion") or 0.0),
            source_chunk_ids=[int(value) for value in result.get("source_chunk_ids") or []],
            rationale=reasoning,
        )
    except (TypeError, ValueError):
        return None


async def run_task_hours_recovery_agent(
    tasks: list[AgentTaskRef],
    *,
    client: Any,
    model: str,
    backend: BudgetSearchBackend,
    consensus_fn: ConsensusFn,
    reasoning_effort: str = "medium",
    max_iterations: int = 10,
) -> AgentTaskHoursRun:
    """Re-busca las tareas marcadas y devuelve las horas recuperadas + la traza.

    ``client``, ``backend`` y ``consensus_fn`` se inyectan: esta capa no resuelve
    infraestructura ni importa ``generation.rag``.
    """
    iteration_budget = max(1, max_iterations)
    trace = AgentTrace(model=model)
    step_number = 0
    iterations = 0
    stopped_reason = "completed"
    recovered: dict[tuple[str, str], AgentTaskDerivation] = {}

    if not tasks:
        trace.iterations = 0
        trace.stopped_reason = "completed"
        return AgentTaskHoursRun(
            recovered=[],
            trace=trace,
            iterations=0,
            stopped_reason="completed",
        )

    log.info(
        "agent.recovery.start",
        model=model,
        effort=reasoning_effort,
        flagged=len(tasks),
        max_iterations=iteration_budget,
    )
    response = await client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": _user_message(tasks)}],
        tools=RECOVERY_TOOL_SCHEMAS,
        reasoning={"effort": reasoning_effort, "summary": "auto"},
        store=True,
    )
    iterations = 1

    while True:
        calls = function_calls(response.output)
        if not calls:
            break
        if iterations >= iteration_budget:
            stopped_reason = "max_iterations"
            log.warning("agent.recovery.max_iterations", iterations=iterations)
            break

        reasoning = summarize_reasoning(response.output)
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
                    result = await dispatch_tool(
                        tool_name,
                        raw_args,
                        backend=backend,
                        consensus_fn=consensus_fn,
                    )
                except Exception as exc:
                    log.warning("agent.tool.error", tool=tool_name, error=str(exc)[:200])
                    result = {"error": f"{type(exc).__name__}: {exc}"}

            if tool_name == "derive_task_hours":
                derivation = _parse_derivation(result, step_reasoning)
                if derivation is not None:
                    recovered[(derivation.module, derivation.task)] = derivation

            trace.steps.append(
                AgentStep(
                    step=step_number,
                    reasoning_summary=step_reasoning,
                    tool=tool_name,
                    tool_args=raw_args,
                    observation=observation(result),
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
            tools=RECOVERY_TOOL_SCHEMAS,
            reasoning={"effort": reasoning_effort, "summary": "auto"},
            store=True,
        )
        iterations += 1

    trace.iterations = iterations
    trace.stopped_reason = stopped_reason  # type: ignore[assignment]
    derivations = list(recovered.values())
    log.info(
        "agent.recovery.done",
        iterations=iterations,
        steps=len(trace.steps),
        recovered=len(derivations),
        flagged=len(tasks),
        stopped_reason=stopped_reason,
    )
    return AgentTaskHoursRun(
        recovered=derivations,
        trace=trace,
        iterations=iterations,
        stopped_reason=stopped_reason,  # type: ignore[arg-type]
    )
