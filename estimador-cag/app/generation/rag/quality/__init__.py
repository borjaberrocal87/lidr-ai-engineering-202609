"""Calidad de la generación RAG (Sesión 11, directo).

Tres capas independientes y conmutables que se aplican alrededor de la
generación grounded:

* :mod:`augmentation` — mejora la ENTRADA (comprime y reordena las fuentes).
* :mod:`hallucination` — audita la SALIDA (¿la cifra está respaldada por la
  evidencia citada, o es una alucinación vestida de cita?).
* :mod:`synthesis` — expresa la INCERTIDUMBRE de las horas (un rango cuando los
  análogos históricos se contradicen, en vez de un punto medio engañoso).

Cada una se puede medir en aislamiento: son palancas, no un paquete cerrado.
"""

from __future__ import annotations

from app.generation.rag.quality.augmentation import (
    AugmentationPolicy,
    apply_augmentation,
    compress_for_context,
    key_lines,
    reorder_strong_edges,
)
from app.generation.rag.quality.hallucination import (
    HallucinationGate,
    combine_line_gate,
    evidence_ceiling_days,
    recorded_hours,
)
from app.generation.rag.quality.synthesis import (
    deterministic_reason,
    is_contradiction,
    synthesize_hour_range,
)

__all__ = [
    "AugmentationPolicy",
    "HallucinationGate",
    "apply_augmentation",
    "combine_line_gate",
    "compress_for_context",
    "deterministic_reason",
    "evidence_ceiling_days",
    "is_contradiction",
    "key_lines",
    "recorded_hours",
    "reorder_strong_edges",
    "synthesize_hour_range",
]
