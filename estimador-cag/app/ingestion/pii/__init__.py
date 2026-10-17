"""Capa de pseudonimización conforme al RGPD.

Tres responsabilidades, tres módulos:

* ``analyzer`` — inicializa el ``AnalyzerEngine`` de Presidio contra
  ``es_core_news_md``. Sin un modelo español explícito, Presidio usa inglés por
  defecto y omite en silencio los nombres hispanos, un momento didáctico en la
  sesión en vivo.

* ``recognizers`` — :class:`PatternRecognizer`s personalizados para los
  identificadores propios del dominio del proyecto (``BUDGET-YYYY-NNNN`` y
  ``CLI-NNNN``).

* ``pseudonymizer`` — el :class:`ConsistentPseudonymizer`. Mismo valor original
  → mismo seudónimo, entre documentos y entre ejecuciones. Respaldado por un
  :class:`MappingStore` persistente, de modo que la operación es reversible
  (art. 17).
"""

from app.ingestion.pii.analyzer import build_analyzer
from app.ingestion.pii.mapping_store import (
    InMemoryMappingStore,
    MappingStore,
    PostgresMappingStore,
)
from app.ingestion.pii.pseudonymizer import (
    ConsistentPseudonymizer,
    PseudonymizationResult,
)
from app.ingestion.pii.recognizers import (
    BudgetIdRecognizer,
    ClientCodeRecognizer,
)

__all__ = [
    "BudgetIdRecognizer",
    "ClientCodeRecognizer",
    "ConsistentPseudonymizer",
    "InMemoryMappingStore",
    "MappingStore",
    "PostgresMappingStore",
    "PseudonymizationResult",
    "build_analyzer",
]
