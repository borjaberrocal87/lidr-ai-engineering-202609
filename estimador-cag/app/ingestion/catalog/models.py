"""Modelos Pydantic v2 del catálogo de fuentes de datos.

El catálogo es el contrato entre *lo que existe en el mundo real* y *lo que el
pipeline de ingesta se toma como fiable*. Hay tres decisiones válidas:

* ``include`` — la fuente entra en el pipeline.
* ``review`` — problemas conocidos, pendiente del visto bueno del propietario.
  Se mantiene en disco, no se ingiere. Conserva la categoría intermedia en vez
  de colapsarla a un binario include/exclude (elección disciplinar intencionada).
* ``exclude`` — fuera explícitamente. Un ``decision_reason`` es obligatorio:
  excluir una fuente es una decisión defendible, no una omisión silenciosa.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CatalogDecision(StrEnum):
    INCLUDE = "include"
    REVIEW = "review"
    EXCLUDE = "exclude"


class QualityScore(BaseModel):
    """Puntuación por dimensión en una escala 1..5, anclada en hechos concretos."""

    model_config = ConfigDict(extra="forbid")

    completeness: int = Field(ge=1, le=5)
    consistency: int = Field(ge=1, le=5)
    actuality: int = Field(ge=1, le=5)
    reliability: int = Field(ge=1, le=5)


class Sensitivity(BaseModel):
    """Si la fuente contiene PII y de qué tipos.

    Lo usa aguas abajo el orquestador para decidir si enruta los documentos por
    el paso de pseudonimización antes de entregarlos para indexación.
    """

    model_config = ConfigDict(extra="forbid")

    has_pii: bool
    pii_flags: list[str] = Field(default_factory=list)
    access_level: Literal["public", "internal", "confidential"] = "internal"


class CatalogSource(BaseModel):
    """Una única fuente de datos auditada."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    location: str  # ruta o URL — la resuelve el loader, no este modelo
    owners: list[str] = Field(default_factory=list)
    format: Literal["json", "txt", "xlsx", "docx", "pdf", "csv"]
    volume_estimate: str = ""  # formato libre, p. ej. "≈ 6 ficheros, 250 KB"
    refresh_declared: str = ""  # lo que dice el propietario: "semanal"
    refresh_observed: str = ""  # lo que midió el inspector
    quality: QualityScore
    sensitivity: Sensitivity
    lineage: list[str] = Field(default_factory=list)
    decision: CatalogDecision
    decision_reason: str = ""
    last_audited: datetime | None = None

    @field_validator("name")
    @classmethod
    def _name_is_snake_case(cls, value: str) -> str:
        if not value or not all(c.islower() or c.isdigit() or c == "_" for c in value):
            raise ValueError(
                "CatalogSource.name must be lowercase snake_case "
                "(used as identifier in the ingestion endpoint)"
            )
        return value

    @model_validator(mode="after")
    def _decision_requires_reason_when_not_include(self) -> CatalogSource:
        if self.decision is not CatalogDecision.INCLUDE and not self.decision_reason:
            raise ValueError(f"decision={self.decision.value} requires a non-empty decision_reason")
        return self


class DataCatalog(BaseModel):
    """El catálogo completo. Versionado en git junto al código."""

    model_config = ConfigDict(extra="forbid")

    version: str
    description: str = ""
    sources: list[CatalogSource]

    @field_validator("sources")
    @classmethod
    def _names_are_unique(cls, sources: list[CatalogSource]) -> list[CatalogSource]:
        seen: set[str] = set()
        for src in sources:
            if src.name in seen:
                raise ValueError(f"Duplicate source name in catalog: {src.name}")
            seen.add(src.name)
        return sources

    def included_sources(self) -> list[CatalogSource]:
        return [s for s in self.sources if s.decision is CatalogDecision.INCLUDE]

    def find(self, name: str) -> CatalogSource | None:
        for src in self.sources:
            if src.name == name:
                return src
        return None
