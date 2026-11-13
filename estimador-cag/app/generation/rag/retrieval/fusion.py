"""Reciprocal Rank Fusion (RRF) — combine independent rankings into one.

RRF fuses rankings by POSITION, not by score, which is exactly what we want when
the two branches produce incomparable scores: cosine distance (lower = better)
from the vector branch and ``ts_rank_cd`` (higher = better) from the lexical one.
Normalising those onto a shared scale is fiddly and corpus-dependent; RRF
sidesteps it by only trusting the order each branch puts things in.

The score of a document is ``sum(1 / (k + rank))`` over every ranking it appears
in (rank is 0-based here). The smoothing constant ``k`` damps the contribution of
top positions: a large ``k`` flattens the curve so a document must rank well in
*both* branches to win, while a small ``k`` lets a single #1 dominate. 60 is the
value from the original Cormack et al. paper and a sane default.

Pure and synchronous on purpose — trivially unit-testable, no I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import zip_longest
from typing import Any

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[int, float]]:
    """Fuse several id rankings into one, best first.

    Parameters
    ----------
    rankings:
        One list of chunk ids per branch, each already ordered best→worst. A
        branch that returned nothing contributes an empty list (it just adds no
        score). Duplicate ids within a single ranking are ignored after their
        first (best) occurrence.
    k:
        RRF smoothing constant (see module docstring). Must be positive.

    Returns
    -------
    list[tuple[int, float]]
        ``(chunk_id, fused_score)`` sorted by score descending. Ties break by
        ascending id so the output is deterministic.
    """
    if k <= 0:
        raise ValueError("RRF smoothing constant k must be positive")

    scores: dict[int, float] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for position, chunk_id in enumerate(ranking):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def round_robin_merge(
    ranked_lists: list[list[Any]], *, key: Callable[[Any], Any] = lambda item: item
) -> list[Any]:
    """Intercala varias listas ordenadas posición a posición, deduplicando.

    Es la fusión de la DESCOMPOSICIÓN: cada sub-query cubre un tema distinto y
    ninguna debe desaparecer porque otra tenga más consenso. RRF premia el
    acuerdo; round-robin premia la cobertura. El primer puesto de cada lista
    entra antes que el segundo de cualquiera.

    ``key`` extrae la identidad para deduplicar. Entre fuentes los ids de BD
    pueden colisionar (cada tabla tiene su propia secuencia), así que el
    llamante pasa una clave compuesta ``(collection, id)``.
    """
    seen: set[Any] = set()
    merged: list[Any] = []
    for column in zip_longest(*ranked_lists):
        for item in column:
            if item is None:
                continue
            identity = key(item)
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(item)
    return merged
