"""Errores del dominio de ingesta (Sesión 6).

Viven en un módulo propio para que el caso de uso (`orchestrator`) pueda señalar
fuentes no ingeribles sin depender de la capa HTTP. El router traduce estos
errores a códigos de estado en un único punto (`create_ingestion_run`), de modo
que la regla "una fuente no incluida no se ingiere" está en un solo lugar.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base de los errores del subsistema de ingesta."""


class UnknownSource(IngestionError):
    """La fuente no existe en el catálogo."""

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name
        super().__init__(f"source {source_name!r} not found in catalog")


class SourceNotIncluded(IngestionError):
    """La fuente existe pero su decisión no es `include`."""

    def __init__(self, source_name: str, decision: str, decision_reason: str = "") -> None:
        self.source_name = source_name
        self.decision = decision
        self.decision_reason = decision_reason
        super().__init__(
            f"source {source_name!r} has decision={decision!r}; "
            f"only 'include' sources can be ingested"
        )
