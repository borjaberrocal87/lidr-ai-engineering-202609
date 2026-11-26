"""La traza del agente (razona → actúa → observa) como contrato compartido (S12).

Vive en ``domain/schemas`` — y NO dentro de ``generation/agentic`` — a propósito:
la traza es un contrato de auditoría que cruza hermanos de ``generation``. El
bucle agentic la PRODUCE, pero los schemas RAG (``GroundedGenerationOutcome`` /
``TaskHoursResult``) la TRANSPORTAN para que el wizard la pinte junto a la
estructura y las horas. Ambos hermanos pueden importar ``domain/schemas``, así
que aparcarla aquí evita que ``rag`` importe ``agentic`` (un salto entre
hermanos prohibido; ver ``tests/test_layering.py``).
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

StoppedReason = Literal["completed", "max_iterations", "no_final_report"]

# Un modelo puede colar un carácter de control dentro de un escape unicode
# malformado. Se limpia solo al renderizar la traza para que el .txt quede
# legible sin tocar los argumentos que usa el dispatch.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class AgentStep(BaseModel):
    """Un paso razonamiento → acción → observación del bucle."""

    step: int = Field(ge=1)
    reasoning_summary: str | None = Field(
        default=None,
        description="Reasoning summary emitted by the Responses API for this turn.",
    )
    tool: str = Field(description="Invoked tool name.")
    tool_args: dict[str, Any] = Field(description="Arguments the model passed.")
    observation: str = Field(description="Human-readable summary of the tool result.")


class AgentTrace(BaseModel):
    """Registro ordenado de la ejecución, para auditar y para el entregable."""

    steps: list[AgentStep] = Field(default_factory=list)
    model: str | None = None
    iterations: int | None = None
    stopped_reason: StoppedReason | None = None

    def render(self) -> str:
        """Render the trace in the ``STEP N`` console format required by the exercise."""
        header = ["=" * 78, "AGENT TRACE", "=" * 78]
        meta = []
        if self.model:
            meta.append(f"model={self.model}")
        if self.iterations is not None:
            meta.append(f"iterations={self.iterations}")
        if self.stopped_reason:
            meta.append(f"stopped={self.stopped_reason}")
        if meta:
            header.append("  " + "   ".join(meta))
            header.append("")

        if not self.steps:
            return "\n".join([*header, "(no tool steps — the agent answered without tools)"])

        blocks: list[str] = []
        for step in self.steps:
            reasoning = step.reasoning_summary or "(no reasoning summary emitted)"
            args = _CONTROL_CHARS.sub(
                "",
                json.dumps(step.tool_args, ensure_ascii=False, default=str),
            )
            blocks.append(
                f"STEP {step.step}\n"
                f"  reasoning:   {reasoning}\n"
                f"  action:      {step.tool}({args})\n"
                f"  observation: {step.observation}"
            )
        return "\n".join([*header, "\n\n".join(blocks)])
