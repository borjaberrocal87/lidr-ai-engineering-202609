"""Comprobaciones post-generación de estimaciones grounded (Sesión 9).

Dos guardas independientes corren tras el LLM:

* :func:`find_fabricated_citations` — cada ``source_id`` citado debe
  corresponder a un chunk realmente recuperado. Los ids inventados son el fallo
  clásico de grounding y disparan un reintento correctivo en el orquestador.
* :func:`is_coherent_estimate` — el nivel ``insufficient`` tiene una forma
  estricta (sin números y con explicación); violarla es una respuesta mal
  formada, no una estimación válida.
"""

from __future__ import annotations

from app.generation.rag.schemas import GroundedEstimate, RetrievedChunk


def find_fabricated_citations(
    estimate: GroundedEstimate,
    retrieved_chunks: list[RetrievedChunk],
) -> list[int]:
    """Devuelve los source ids citados que nunca se recuperaron (inventados).

    Revisa tanto las citas de nivel superior (``sources``) como las de cada
    tarea (``modules[].tasks[].sources``). Una lista vacía significa que todas
    las citas son válidas (incluido el caso de una estimación que no cita nada).
    """
    valid_ids = {chunk.id for chunk in retrieved_chunks}

    cited_ids: set[int] = {citation.source_id for citation in estimate.sources}
    for module in estimate.modules:
        for task in module.tasks:
            cited_ids.update(task.sources)

    return sorted(cited_ids - valid_ids)


def is_coherent_estimate(estimate: GroundedEstimate) -> bool:
    """Comprueba que el nivel de confianza encaja con el contenido.

    Cuando ``confidence == "insufficient"``: ambos totales numéricos deben ser
    ``None``, ``modules`` debe estar vacío y la explicación de insuficiencia
    debe ser no vacía. Cualquier otro nivel se considera coherente aquí; las
    comprobaciones numéricas pertenecen al esquema y a las reglas de negocio.
    """
    if estimate.confidence != "insufficient":
        return True
    return (
        estimate.total_engineer_days is None
        and estimate.duration_weeks is None
        and not estimate.modules
        and bool(estimate.insufficient_context_explanation)
    )
