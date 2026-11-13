"""Vigencia temporal: penaliza los finalistas envejecidos (Sesión 10).

Un embedding codifica lo que un chunk DICE, no cuándo se escribió ni si sigue
siendo cierto. El presupuesto de AngularJS de 2019 puede ser semánticamente
perfecto y operativamente peligroso. La vigencia es la contraparte SUAVE del
filtro duro por fecha: en vez de excluir los chunks antiguos, multiplica su
relevancia por un decaimiento exponencial para que la recencia desempate sin
silenciar la historia.

``weight = 0.5 ** (edad_días / vida_media_días)`` — un chunk con exactamente una
vida media conserva la mitad de su score. La vida media por defecto es generosa
(≈900 días ≈ 2,5 años) porque los presupuestos envejecen despacio.

Se aplica AL FINAL, sobre los finalistas, DESPUÉS del reranking: el
cross-encoder decide relevancia y luego la vigencia solo ajusta el orden final.
Nunca antes del reranking, o la recencia distorsionaría el conjunto de
candidatos que ve el reranker.

Precondición: el score base debe ser NO NEGATIVO (la vigencia es un multiplicador
0..1; un score negativo se acercaría a cero y invertiría el orden). El pipeline
multifuente se encarga de mapear los logits del reranker por una sigmoide antes
de llamar aquí.

Puro y síncrono: sin I/O y con la fecha de referencia inyectada, para que los
tests sean deterministas y el módulo no llame al reloj.
"""

from __future__ import annotations

from datetime import date

from app.generation.rag.schemas import RetrievedChunk


def freshness_weight(age_days: float, half_life_days: float) -> float:
    """Multiplicador de decaimiento en ``(0, 1]``.

    Una vida media no positiva desactiva el decaimiento (peso 1.0). Un chunk con
    fecha futura o sin edad también pesa 1.0 (lo fresco no se penaliza).
    """
    if half_life_days <= 0 or age_days <= 0:
        return 1.0
    return float(0.5 ** (age_days / half_life_days))


def apply_freshness(
    scored_chunks: list[tuple[RetrievedChunk, float]],
    *,
    half_life_days: float,
    reference_date: date,
) -> list[tuple[RetrievedChunk, float]]:
    """Re-pondera ``(chunk, score_base)`` por recencia y reordena.

    Cada score base (no negativo; ver docstring del módulo) se multiplica por el
    peso de vigencia del chunk (desde su ``document_date``; sin fecha, peso 1.0)
    y la lista se reordena de mejor a peor. Devuelve pares
    ``(chunk, score_ajustado)``.
    """
    if not scored_chunks:
        return []

    weighted: list[tuple[RetrievedChunk, float]] = []
    for chunk, base in scored_chunks:
        if chunk.document_date is not None:
            age_days = (reference_date - chunk.document_date).days
            weight = freshness_weight(age_days, half_life_days)
        else:
            weight = 1.0
        weighted.append((chunk, base * weight))

    weighted.sort(key=lambda pair: pair[1], reverse=True)
    return weighted
