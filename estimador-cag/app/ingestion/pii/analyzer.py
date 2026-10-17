"""Factoría de AnalyzerEngine de Presidio configurada para texto en español.

La configuración por defecto de Presidio viene con modelos spaCy en inglés.
Sobre nombres hispanos ("Laura Fernández", "Javier Romero") devuelve en
silencio *cero* entidades PERSON — un modo de fallo mucho peor que un falso
positivo porque pasa desapercibido. Este módulo es el único lugar que conecta
Presidio con ``es_core_news_md``.

La sesión en vivo usa este módulo mostrando primero el fallo con la
configuración por defecto y sustituyéndola DESPUÉS por esta.
"""

from __future__ import annotations

from functools import lru_cache

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from app.config import get_settings
from app.ingestion.pii.recognizers import (
    BudgetIdRecognizer,
    ClientCodeRecognizer,
)


@lru_cache
def build_analyzer() -> AnalyzerEngine:
    """Devuelve un AnalyzerEngine de Presidio conectado al modelo español configurado.

    Es un singleton a propósito: la carga del modelo spaCy es lenta (~300 ms) y
    Presidio es seguro entre hilos para análisis de solo lectura. Los
    reconocedores personalizados (BUDGET_ID, CLIENT_CODE) se registran de forma
    anticipada para que quien llame no tenga que acordarse de hacerlo.
    """
    settings = get_settings()
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "es", "model_name": settings.presidio_spacy_model}],
        }
    )
    engine = AnalyzerEngine(
        nlp_engine=provider.create_engine(),
        supported_languages=["es"],
    )
    engine.registry.add_recognizer(BudgetIdRecognizer())
    engine.registry.add_recognizer(ClientCodeRecognizer())
    return engine
