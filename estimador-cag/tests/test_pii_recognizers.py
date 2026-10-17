"""Pruebas para los reconocedores personalizados de Presidio (solo regex, sin cargar NLP)."""

from __future__ import annotations

from app.ingestion.pii.recognizers import BudgetIdRecognizer, ClientCodeRecognizer


def test_budget_id_recognizer_matches_canonical_form():
    rec = BudgetIdRecognizer()
    # Se puede llamar directamente a PatternRecognizer.analyze; nlp_artifacts puede ser None.
    results = rec.analyze(
        text="Estamos revisando BUDGET-2024-0001 hoy.",
        entities=["BUDGET_ID"],
        nlp_artifacts=None,
    )
    assert len(results) == 1
    assert results[0].entity_type == "BUDGET_ID"
    assert results[0].score >= 0.9


def test_client_code_recognizer_matches():
    rec = ClientCodeRecognizer()
    results = rec.analyze(
        text="Cliente CLI-0042 con código activo.",
        entities=["CLIENT_CODE"],
        nlp_artifacts=None,
    )
    assert len(results) == 1
    assert results[0].entity_type == "CLIENT_CODE"


def test_client_code_does_not_match_wrong_shape():
    rec = ClientCodeRecognizer()
    # CLI-42 (solo dos dígitos) no debería coincidir.
    assert (
        rec.analyze(
            text="Cliente CLI-42 desconocido.",
            entities=["CLIENT_CODE"],
            nlp_artifacts=None,
        )
        == []
    )


def test_budget_id_does_not_match_partial():
    rec = BudgetIdRecognizer()
    assert (
        rec.analyze(
            text="Algo así como BUDGET-2024 sin código completo.",
            entities=["BUDGET_ID"],
            nlp_artifacts=None,
        )
        == []
    )
