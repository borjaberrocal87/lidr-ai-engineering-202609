"""Modelos del pipeline de embeddings.

Entrada: presupuestos históricos normalizados (un presupuesto con lista de
componentes). Salida: chunks listos para embeber y, una vez vectorizados, los
vectores con sus estadísticas agregadas.
"""

from __future__ import annotations

from enum import StrEnum

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
    """Payload de entrada de ``POST /api/v1/embeddings/ingest``."""

    model_config = ConfigDict(extra="forbid")

    budgets: list[Budget] = Field(
        ..., min_length=1, description="Presupuestos a chunquear y embeber."
    )


class IngestStats(BaseModel):
    """Contadores agregados que acompañan a una respuesta de ingesta."""

    total_budgets: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    estimated_cost_usd: float = Field(..., ge=0.0)


class IngestResponse(BaseModel):
    """Respuesta de ``POST /api/v1/embeddings/ingest``."""

    chunks: list[EmbeddedChunk]
    stats: IngestStats
