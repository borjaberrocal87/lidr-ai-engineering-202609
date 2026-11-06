"""Aumentación: ensamblar los chunks recuperados en un contexto delimitado.

Dos anti-patrones que este módulo existe para evitar:

* ``"\\n\\n".join(chunk.content for chunk in chunks)`` — el modelo no sabe dónde
  acaba una fuente y empieza la siguiente, ni qué id citar. Cada chunk se
  envuelve en un elemento ``<source>`` con su id y metadata para que el
  generador pueda fundamentar cada afirmación en una fuente citable.
* Partir un chunk por la mitad cuando el contexto desborda. El presupuesto se
  recorta descartando chunks completos desde la cola (los menos relevantes):
  media línea de presupuesto es peor que una línea menos.
"""

from __future__ import annotations

from typing import Protocol

from app.generation.rag.schemas import RetrievedChunk


class TokenEncoder(Protocol):
    """Contrato mínimo de un encoder de tokens (``tiktoken``)."""

    def encode(self, text: str) -> list[int]: ...


def render_source(chunk: RetrievedChunk) -> str:
    """Renderiza un chunk como elemento ``<source>`` autodescriptivo."""
    return (
        f'<source id="{chunk.id}" sector="{chunk.sector}" '
        f'project_year="{chunk.project_year}" chunk_type="{chunk.chunk_type}" '
        f'distance="{chunk.distance:.4f}">\n'
        f"{chunk.content}\n"
        f"</source>"
    )


def assemble_context(chunks: list[RetrievedChunk]) -> str:
    """Construye el bloque XML que se envía al generador.

    Los chunks llegan ordenados por distancia ascendente (el retriever ya los
    devuelve así); cada uno se emite como un ``<source>``.
    """
    return "\n".join(render_source(chunk) for chunk in chunks)


def fit_chunks_to_budget(
    chunks: list[RetrievedChunk],
    max_context_tokens: int,
    encoder: TokenEncoder,
) -> list[RetrievedChunk]:
    """Conserva los chunks iniciales que quepan en ``max_context_tokens``.

    Los tokens se cuentan sobre el chunk **ya envuelto en su XML**, no solo
    sobre el contenido: los delimitadores son tokens reales que verá el modelo.
    Se mantiene el orden y ningún chunk se parte.
    """
    kept: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        cost = len(encoder.encode(render_source(chunk)))
        if used + cost > max_context_tokens:
            break
        kept.append(chunk)
        used += cost
    return kept
