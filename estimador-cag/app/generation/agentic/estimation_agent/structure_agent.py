"""Fase 1 del agente: propone la estructura módulo→tarea (Sesión 12).

Un único ``responses.parse`` razonado: el agente lee el brief (ya destilado por el
conductor) y lo descompone en módulos y tareas. **Sin tools y sin horas**: groundar
la estructura en presupuestos empobrecía el árbol (decisión de la Sesión 10); las
horas las pone después la fase 2, tarea a tarea.

Sigue siendo la excepción deliberada a ``LLMWrapper``: hablar con la Responses
API cruda es lo que permite capturar la traza igual que en el resto del bucle.

Esta capa recibe el brief como TEXTO: no importa ``generation.rag`` (hermano
prohibido); el conductor es quien conoce ``ProjectBrief`` y compone el mensaje.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.domain.schemas.agent_trace import AgentTrace
from app.generation.agentic.estimation_agent.schemas import (
    AgentStructure,
    AgentStructureRun,
)

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """\
You are an estimation agent for a software consultancy. You receive the \
structured brief of a discovery meeting and must decompose it into the functional \
modules and concrete tasks required to deliver it.

Rules:
1. Propose a module→task tree that covers the whole project. Real projects \
usually mix pieces of different nature (a business backend, an ERP or third-party \
integration, a mobile app, an analytics dashboard): keep unrelated pieces in \
separate modules instead of merging them.
2. Each task must describe a concrete, estimable unit of work, not a whole phase.
3. Do NOT assign hours: you do not have historical data yet. The hours are derived \
afterwards, task by task, against the historical corpus.
4. Do NOT invent requirements that are not in the brief. Everything you have to \
assume goes into `assumptions`.
5. If the brief is too vague to decompose, say so in \
`insufficient_context_explanation` and keep the tree minimal.
6. Choose a confidence level that reflects how concrete the brief is.
"""


async def run_structure_agent(
    brief_text: str,
    *,
    client: Any,
    model: str,
    reasoning_effort: str = "low",
) -> AgentStructureRun:
    """Pide al agente la estructura del brief y devuelve también su traza.

    ``client`` es un ``AsyncOpenAI`` inyectado por el router; esta capa no
    resuelve infraestructura.
    """
    log.info("agent.structure.start", model=model, effort=reasoning_effort)
    parsed = await client.responses.parse(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=[{"role": "user", "content": brief_text}],
        text_format=AgentStructure,
        reasoning={"effort": reasoning_effort, "summary": "auto"},
        store=True,
    )
    structure = parsed.output_parsed
    if structure is None:
        raise ValueError("the structure agent returned no parseable structure")

    trace = AgentTrace(model=model, iterations=1, stopped_reason="completed")
    log.info(
        "agent.structure.done",
        modules=len(structure.modules),
        tasks=sum(len(module.tasks) for module in structure.modules),
        confidence=structure.confidence,
    )
    return AgentStructureRun(
        structure=structure,
        trace=trace,
        iterations=1,
        stopped_reason="completed",
    )
