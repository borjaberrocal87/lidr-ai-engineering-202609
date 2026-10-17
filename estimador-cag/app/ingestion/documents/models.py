"""El modelo canónico ``Document``.

Diseñado plano a propósito: un único sub-objeto ``DocumentMetadata`` contiene los
campos estructurados que necesita cada consumidor; todo lo específico de formato
vive en ``extra: dict[str, Any]``. Esa vía de escape es lo que nos permite
mantener el contrato uniforme entre JSON, TXT, XLSX y formatos futuros sin
inventar sub-tipos por fuente.

El ``id`` es el identificador determinista que asigna el orquestador; debe ser
estable entre re-ingestas para que el índice aguas abajo pueda deduplicar.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str
    source_version: str  # el ``DataCatalog.version`` activo en el momento de la ingesta
    ingested_at: datetime
    lineage: list[str] = Field(default_factory=list)
    sensitivity_pii_flags: list[str] = Field(default_factory=list)
    sensitivity_access_level: str = "internal"
    location: str = ""  # ruta o URL de la que vino el documento
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    metadata: DocumentMetadata
