"""Post-generation checks for grounded estimates (Sessions 9 & 11).

Two independent guards run after the LLM returns a :class:`GroundedEstimate`:

* :func:`verify_citations` (Session 11) — line-level, programmatic citation
  verification. Every cited ``chunk_id`` (per-line :class:`SourceReference` and
  the estimate-global :class:`Citation`) must resolve to a chunk that was
  actually retrieved. Ids that were never in the context are *dangling*: a
  hallucination wearing the costume of rigor. The returned
  :class:`CitationReport` separates correctly grounded lines, dangling
  citations and lines explicitly marked as having no sufficient source data.
* :func:`is_coherent_estimate` — the ``insufficient`` confidence level has a
  strict shape (no numbers, an explanation present); a violation is a malformed
  response, not a valid estimate.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.generation.rag.context_builder import traceable_document_id
from app.generation.rag.schemas import (
    CitationReport,
    GroundedEstimate,
    LineCitation,
    LineCitationStatus,
    RetrievedChunk,
)


def verify_citations(
    estimate: GroundedEstimate,
    retrieved_chunk_ids: set[str],
    *,
    retrieved_documents: Mapping[str, str] | None = None,
) -> CitationReport:
    """Verify every citation against the chunks actually handed to the LLM.

    Walks each estimate line (``modules[].tasks[]``) and checks that every
    ``chunk_id`` it cites is present in ``retrieved_chunk_ids``. When
    ``retrieved_documents`` is provided, the cited ``document_id`` must also
    match the traceable parent of the retrieved chunk; a chunk that exists but
    points to the wrong budget is just as misleading as an invented id.

    The estimate-global ``sources`` (Session 9 :class:`Citation`) are checked
    too, so a fabricated id cannot hide at either level.

    Parameters
    ----------
    estimate:
        The generated estimate to audit.
    retrieved_chunk_ids:
        The ids of the chunks that were actually placed in the context block
        (typically ``{str(chunk.id) for chunk in kept}``).
    retrieved_documents:
        Optional mapping ``chunk_id -> traceable document id`` used to verify
        that each citation names the right historical budget.

    Returns
    -------
    CitationReport
        Per-line statuses plus aggregate counts. ``dangling_citations`` is the
        sorted, de-duplicated set of cited ids that were never retrieved; an
        empty list means every citation is real.
    """
    lines: list[LineCitation] = []
    verified = 0
    dangling: set[str] = set()
    mismatches: set[str] = set()
    grounded_lines = dangling_lines = insufficient_lines = 0

    for module in estimate.modules:
        for task in module.tasks:
            explicit_grounding = "grounded" in task.model_fields_set
            cited = [reference.chunk_id for reference in task.citations]
            if not cited and task.sources:
                cited = [str(source_id) for source_id in task.sources]
            grounded = task.grounded if explicit_grounding else bool(cited)

            line_dangling = [chunk_id for chunk_id in cited if chunk_id not in retrieved_chunk_ids]
            line_mismatches: list[str] = []
            if retrieved_documents is not None:
                for reference in task.citations:
                    expected = retrieved_documents.get(reference.chunk_id)
                    if expected is not None and reference.document_id != expected:
                        line_mismatches.append(
                            f"{reference.chunk_id}: cited document_id="
                            f"{reference.document_id!r} expected={expected!r}"
                        )

            verified += len(cited) - len(line_dangling)
            dangling.update(line_dangling)
            mismatches.update(line_mismatches)

            if not grounded:
                status: LineCitationStatus = "insufficient"
                insufficient_lines += 1
                reason = "no sufficient source data"
            elif not cited:
                status = "dangling"
                dangling_lines += 1
                reason = "grounded line has no citations"
            elif line_dangling or line_mismatches:
                status = "dangling"
                dangling_lines += 1
                reason = "cited chunk not in context or wrong document"
            else:
                status = "grounded"
                grounded_lines += 1
                reason = None

            lines.append(
                LineCitation(
                    module=module.name,
                    component=task.name,
                    status=status,
                    cited_chunk_ids=cited,
                    dangling_chunk_ids=line_dangling,
                    document_mismatches=line_mismatches,
                    reason=reason,
                )
            )

    # Estimate-global citations (the coarse Session 9 layer) are verified too:
    # a fabricated id there is just as much a grounding failure.
    global_dangling: list[str] = []
    for citation in estimate.sources:
        chunk_id = str(citation.source_id)
        if chunk_id in retrieved_chunk_ids:
            verified += 1
        else:
            global_dangling.append(chunk_id)
    dangling.update(global_dangling)

    return CitationReport(
        total_lines=len(lines),
        grounded_lines=grounded_lines,
        dangling_lines=dangling_lines,
        insufficient_lines=insufficient_lines,
        verified_citations=verified,
        dangling_citations=sorted(dangling),
        dangling_global_citations=sorted(global_dangling),
        document_mismatches=sorted(mismatches),
        lines=lines,
    )


def verify_citations_for_chunks(
    estimate: GroundedEstimate,
    retrieved_chunks: list[RetrievedChunk],
) -> CitationReport:
    """Convenience wrapper: derive ids and traceable documents from the chunks."""
    return verify_citations(
        estimate,
        {str(chunk.id) for chunk in retrieved_chunks},
        retrieved_documents={
            str(chunk.id): traceable_document_id(chunk) for chunk in retrieved_chunks
        },
    )


def find_fabricated_citations(
    estimate: GroundedEstimate,
    retrieved_chunks: list[RetrievedChunk],
) -> list[int]:
    """Legacy helper: return cited int ids that were never retrieved.

    Kept for the Session 9 tests and any external consumer. New code should use
    :func:`verify_citations_for_chunks` and read the full report.
    """
    report = verify_citations_for_chunks(estimate, retrieved_chunks)
    return sorted(
        int(chunk_id) for chunk_id in report.dangling_citations if chunk_id.lstrip("-").isdigit()
    )


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
