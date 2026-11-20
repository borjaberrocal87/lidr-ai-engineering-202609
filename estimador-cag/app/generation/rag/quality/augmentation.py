"""Augmentación del contexto recuperado ANTES de la generación (Sesión 11).

El retrieval entrega chunks ordenados por relevancia; el presupuesto de tokens
los recorta desde la cola. Eso deja dos ineficiencias que este módulo corrige con
dos pases deterministas (sin LLM, sin coste, conmutables por separado para medir
cada uno injerto en las evals):

* **Compresión extractiva** — un chunk de presupuesto es sobre todo prosa que el
  generador no necesita; lo que SÍ debe citar es el nombre del componente, su id
  y sus horas. Conservamos las líneas que llevan una cifra o un token con forma
  de id y descartamos el relleno: el mismo presupuesto compra más fuentes
  distintas.
* **Reordenado por extremos** — contramedida al *lost in the middle*: los modelos
  atienden mejor al principio y al final de un contexto largo. Con los chunks
  ordenados de mejor a peor, colocamos los pares en la cabeza (ascendente) y los
  impares en la cola, de modo que la 2ª mejor fuente queda la última y las más
  débiles se entierran en el centro.

Los ids no se tocan: la verificación de citas posterior sigue funcionando igual.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.generation.rag.schemas import RetrievedChunk


@dataclass(frozen=True, slots=True)
class AugmentationPolicy:
    """Palancas de la augmentación, conmutables para medirlas por separado."""

    compress: bool = True
    reorder: bool = True


# Líneas que anclan la cita aunque no lleven cifras: el nombre del componente y
# las horas son justo lo que el generador debe poder copiar.
_ANCHOR_PREFIXES = (
    "component:",
    "module:",
    "task:",
    "componente:",
    "módulo:",
    "tarea:",
    "estimated hours:",
    "horas estimadas:",
    "ref:",
    "ref::",
)


def _is_anchor(line: str) -> bool:
    lowered = line.lower()
    return lowered.startswith(_ANCHOR_PREFIXES) or "::" in line or any(ch.isdigit() for ch in line)


def key_lines(content: str) -> list[str]:
    """Líneas que anclan la evidencia: cifras, ids o etiquetas de componente.

    Si ninguna línea cumple el criterio, degrada a la primera línea no vacía
    para no comprimir un chunk a la nada. Un contenido vacío se queda vacío.
    """
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    kept = [line for line in lines if _is_anchor(line)]
    return kept or lines[:1]


def compress_for_context(chunk: RetrievedChunk) -> RetrievedChunk:
    """Copia del chunk cuyo contenido se reduce a sus líneas clave."""
    return chunk.model_copy(update={"content": "\n".join(key_lines(chunk.content))})


def reorder_strong_edges(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reordena una lista ordenada por relevancia para cargar los extremos.

    Los índices pares (1ª, 3ª, … mejor) van a la cabeza en orden; los impares
    (2ª, 4ª, …) a la cola en orden inverso, de modo que la 2ª mejor cierra el
    contexto y las peores quedan en el medio. Con menos de tres chunks no hay
    "medio" que enterrar: se devuelve el orden original.
    """
    if len(chunks) < 3:
        return list(chunks)
    head = chunks[0::2]
    tail = chunks[1::2]
    return head + tail[::-1]


def apply_augmentation(
    chunks: list[RetrievedChunk],
    policy: AugmentationPolicy | None = None,
) -> list[RetrievedChunk]:
    """Aplica la política de augmentación: comprimir y luego reordenar.

    El orden importa: comprimir primero abarata los tokens y reordenar después
    posiciona el resultado ya recortado.
    """
    resolved = policy or AugmentationPolicy()
    out = [compress_for_context(chunk) for chunk in chunks] if resolved.compress else list(chunks)
    if resolved.reorder:
        out = reorder_strong_edges(out)
    return out
