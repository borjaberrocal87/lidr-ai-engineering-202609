"""Piezas compartidas por las dos fases del bucle manual (Sesión 12).

Las tres fases (autónoma, estructura y recuperación) hablan con la Responses API
igual: resumen del reasoning del turno, recogida de ``function_call`` y una
observación legible del resultado de una tool. Estos helpers evitan duplicar esa
lectura defensiva en cada módulo.
"""

from __future__ import annotations

import json
from typing import Any


def summarize_reasoning(output: list[Any]) -> str | None:
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


def function_calls(output: list[Any]) -> list[Any]:
    """Items ``function_call`` de una respuesta, en su orden."""
    return [item for item in output if getattr(item, "type", None) == "function_call"]


def observation(result: dict[str, Any]) -> str:
    """Resumen legible del resultado de una tool para la traza."""
    summary = result.get("summary")
    if summary:
        return str(summary)
    if result.get("error"):
        return str(result["error"])
    return json.dumps(result, ensure_ascii=False)[:200]
