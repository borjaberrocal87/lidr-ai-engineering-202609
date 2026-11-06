"""Modelos del pipeline de embeddings.

Entrada: presupuestos históricos normalizados (un presupuesto con lista de
componentes). Salida: chunks listos para embeber y, una vez vectorizados, los
vectores con sus estadísticas agregadas.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Sector(StrEnum):
    """Sector de negocio del cliente (universo cerrado del sample)."""

    FINANCE = "finance"
    ECOMMERCE = "ecommerce"
    HEALTHCARE = "healthcare"
    INDUSTRIAL = "industrial"


class Complexity(StrEnum):
    """Bucket de complejidad del componente."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClientMetadata(BaseModel):
    """Cliente del presupuesto. Viaja como contexto filtrable; no se embebe."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Nombre de la empresa cliente.")
    sector: Sector = Field(..., description="Sector de negocio del cliente.")
    country: str = Field(..., description="Código de país, p. ej. 'ES'.")


class BudgetComponent(BaseModel):
    """Una línea de un presupuesto histórico."""

    model_config = ConfigDict(extra="forbid")

    component_id: str = Field(
        ..., description="Id estable dentro del presupuesto, p. ej. 'AUTH-001'."
    )
    name: str = Field(..., description="Nombre corto y legible del componente.")
    description: str = Field(..., description="Descripción detallada del trabajo.")
    tech_stack: list[str] = Field(
        default_factory=list, description="Tecnologías implicadas en el componente."
    )
    estimated_hours: int = Field(..., ge=0, description="Horas estimadas para el componente.")
    complexity: Complexity = Field(..., description="Bucket de complejidad.")
    dependencies: list[str] = Field(
        default_factory=list, description="component_id de los que depende este componente."
    )


class Budget(BaseModel):
    """Presupuesto histórico completo."""

    model_config = ConfigDict(extra="forbid")

    budget_id: str = Field(..., description="Id estable del presupuesto, p. ej. 'BUD-2024-014'.")
    client_metadata: ClientMetadata
    project_summary: str = Field(..., description="Resumen en una línea del proyecto.")
    main_technology: str = Field(..., description="Tecnología principal del proyecto.")
    year: int = Field(..., ge=2000, le=2100, description="Año en que se produjo el presupuesto.")
    total_estimated_hours: int = Field(..., ge=0, description="Suma de horas de los componentes.")
    components: list[BudgetComponent] = Field(
        ..., min_length=1, description="Líneas del presupuesto."
    )


class Chunk(BaseModel):
    """Fragmento listo para embeber.

    ``text`` es lo que se manda a la API de embeddings; ``metadata`` lleva los
    campos filtrables que viajan con el chunk pero NO se embeben.
    """

    chunk_id: str = Field(..., description="Id trazable con formato '{budget_id}::{component_id}'.")
    text: str = Field(
        ..., description="Texto embebible: contexto del padre + detalle del componente."
    )
    metadata: dict[str, str | int] = Field(
        default_factory=dict, description="Campos filtrables que no se embeben."
    )
    token_count: int = Field(..., ge=0, description="Tokens de ``text`` según tiktoken.")


class EmbeddedChunk(Chunk):
    """Un :class:`Chunk` con su vector de embedding adjunto."""

    embedding: list[float] = Field(
        ..., description="Vector denso (1536 dims con text-embedding-3-small)."
    )


class IngestRequest(BaseModel):
    """Payload de ``POST /api/v1/embeddings/ingest`` (contrato persistente).

    Una petición = un documento. ``content`` es el JSON completo del
    presupuesto, validado contra :class:`Budget`, de modo que un corpus mal
    formado falla con 422 antes de tocar la base de datos o la API de
    embeddings.
    """

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Procedencia del documento; única por ingesta.",
    )
    document_type: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Familia documental, p. ej. 'historical_budget'.",
    )
    content: Budget = Field(..., description="JSON completo del presupuesto.")


class IngestResponse(BaseModel):
    """Respuesta de ``POST /api/v1/embeddings/ingest``.

    Los vectores ya no viajan por HTTP: viven en pgvector. Solo se devuelven
    identificadores y métricas de la ingesta.
    """

    document_id: int = Field(..., description="Clave primaria del documento persistido.")
    chunks_created: int = Field(..., ge=0, description="Chunks persistidos para este documento.")
    embedding_dimension: int = Field(..., description="Dimensionalidad de los vectores guardados.")
    ingestion_time_ms: int = Field(..., ge=0, description="Tiempo de pared de la ingesta.")


class SearchRequest(BaseModel):
    """Payload de ``POST /api/v1/search``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Consulta semántica en texto libre.")
    k: int = Field(default=5, ge=1, le=50, description="Número de chunks más cercanos.")


class SearchHit(BaseModel):
    """Un chunk rankeado.

    ``chunk_id`` es la clave primaria de la base de datos; el id trazable del
    corpus (``BUD-X::COMP-Y``) viaja dentro de ``metadata``.
    """

    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float = Field(..., description="Distancia coseno (menor = más similar).")
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    """Respuesta de ``POST /api/v1/search``."""

    query: str
    k: int
    search_time_ms: int = Field(..., ge=0)
    results: list[SearchHit]
