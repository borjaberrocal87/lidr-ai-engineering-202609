"""Compatibilidad con el pipeline de la Sesión 7.

El chunker estructural vive ahora en :mod:`app.generation.rag.chunking.structural`
para compartir la interfaz ``Chunker`` y los helpers de render con las
estrategias del comparador. Este módulo mantiene los imports históricos.
"""

from __future__ import annotations

from app.generation.rag.chunking.base import count_tokens
from app.generation.rag.chunking.structural import JSONStructuralChunker

__all__ = ["JSONStructuralChunker", "count_tokens"]
