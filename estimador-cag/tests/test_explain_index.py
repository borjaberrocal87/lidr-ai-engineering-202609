"""Chequeo de los índices HNSW de la Sesión 11 (piezas puras del script)."""

from __future__ import annotations

from scripts.rag_ops.explain_index import (
    TABLES,
    definition_is_aligned,
    expected_index_name,
)


def test_expected_index_name_matches_the_migration() -> None:
    assert expected_index_name("budget_chunks") == "ix_budget_chunks_embedding_hnsw"
    assert TABLES == ("budget_chunks", "transcript_chunks", "technical_doc_chunks")


def test_aligned_definition_accepts_the_session11_index() -> None:
    indexdef = (
        "CREATE INDEX ix_budget_chunks_embedding_hnsw ON public.budget_chunks "
        "USING hnsw (((embedding)::halfvec(1536)) halfvec_cosine_ops) "
        "WITH (m='16', ef_construction='128')"
    )

    assert definition_is_aligned(indexdef) is True


def test_aligned_definition_rejects_full_vector_indexes() -> None:
    full_vector = (
        "CREATE INDEX chunks_embedding_hnsw_idx ON public.budget_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    wrong_ops = (
        "CREATE INDEX ix ON public.budget_chunks "
        "USING hnsw (((embedding)::halfvec(1536)) vector_cosine_ops)"
    )

    assert definition_is_aligned(full_vector) is False
    assert definition_is_aligned(wrong_ops) is False
    assert definition_is_aligned(None) is False
