"""Modelos del pipeline de embeddings.

Entrada: presupuestos históricos normalizados (un presupuesto con lista de
componentes). Salida: chunks listos para embeber y, una vez vectorizados, los
vectores con sus estadísticas agregadas.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

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
    # Sesión 9 — agrupación funcional de la tarea dentro del presupuesto. El
    # corpus task-level la usa para pintar ``Module:`` en el texto embebible.
    module: str | None = Field(default=None, description="Módulo funcional del componente.")


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
    # Sesión 9 — vocabulario de tipos de chunk. El corpus de presupuestos usa
    # ``budget_component``; el corpus task-level, ``historical_task``.
    chunk_type: str = Field(
        default="budget_component",
        min_length=1,
        max_length=50,
        description="Tipo de chunk persistido (columna filtrable).",
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


# --- Sesión 9 — RAG end-to-end sobre transcripciones -------------------------
# La estimación grounded es un contrato distinto del `EstimationResult` de la
# Sesión 4: aquí se trabaja en engineer-days, con módulos → tareas y citas
# obligatorias a los chunks históricos que respaldan cada cifra.

Scale = Literal["small", "medium", "large", "unknown"]
Confidence = Literal["high", "medium", "low", "insufficient"]
Relevance = Literal["primary", "supporting", "tangential"]
Impact = Literal["high", "medium", "low"]


class ProjectBrief(BaseModel):
    """Brief estructurado destilado de una transcripción de reunión.

    Es la salida de la etapa de comprensión: una transcripción tiene digresiones
    y ruido, así que extraemos solo lo que gobierna el retrieval (qué construir,
    con qué tecnología y bajo qué restricciones) en vez de embeber el texto
    crudo.
    """

    functionality: str = Field(description="Resumen funcional del proyecto.")
    technologies: list[str] = Field(default_factory=list)
    sector: str | None = Field(default=None, description="Sector del cliente, si se menciona.")
    scale: Scale = "unknown"
    country: str | None = Field(default=None, description="Código de país, p. ej. 'ES'.")
    regulations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """Un chunk devuelto por el retriever con filtros de metadata.

    ``id`` es la clave primaria del chunk (el ``source id`` que se cita aguas
    abajo). ``sector`` y ``project_year`` se aplanan desde el JSONB del chunk
    para que generador y validador de citas vean una forma estable y tipada.
    """

    id: int
    document_id: int
    content: str
    sector: str | None = None
    project_year: int | None = None
    chunk_type: str
    distance: float = Field(description="Distancia coseno (menor = más similar).")


class RetrievalOutcome(BaseModel):
    """Resultado de la etapa de retrieval."""

    chunks: list[RetrievedChunk]
    low_confidence: bool = Field(
        description="True cuando ningún chunk superó el umbral (soft-fail)."
    )
    candidates_evaluated: int = Field(
        ge=0, description="Chunks puntuados antes de aplicar umbral y límite."
    )


class Citation(BaseModel):
    """Referencia de la estimación a un chunk recuperado."""

    source_id: int = Field(description="Id de BD del chunk citado (RetrievedChunk.id).")
    relevance: Relevance
    used_for: str = Field(description="Qué aportó esta fuente a la estimación.")


class Assumption(BaseModel):
    """Componente de la estimación NO respaldado por ninguna fuente."""

    description: str
    impact: Impact
    rationale: str


class EstimationTask(BaseModel):
    """Una tarea concreta dentro de un módulo funcional, en engineer-days.

    ``sources`` cita los chunks históricos de los que se derivó la tarea; una
    tarea sin análogo histórico se deja sin citar y debe aflorar como
    :class:`Assumption`.
    """

    name: str
    description: str | None = Field(default=None, description="Alcance en una línea.")
    engineer_days: int = Field(ge=0)
    sources: list[int] = Field(default_factory=list, description="Ids de chunk que la respaldan.")


class EstimationModule(BaseModel):
    """Bloque funcional (Auth, Pagos, Datos, Frontend, Infra, QA, PM…) que
    agrupa las tareas concretas necesarias para entregarlo."""

    name: str
    description: str | None = Field(default=None, description="Qué cubre este bloque funcional.")
    tasks: list[EstimationTask] = Field(default_factory=list)


class GroundedEstimate(BaseModel):
    """Estimación grounded producida a partir de presupuestos históricos.

    Basada en engineer-days y con citas obligatorias, distinta del
    ``EstimationResult`` de la Sesión 4 (euros/semanas/fases). El desglose se
    organiza en módulos funcionales, cada uno descompuesto en tareas concretas.
    Cuando el contexto recuperado es insuficiente, ``confidence='insufficient'``
    y los totales numéricos quedan a ``None`` con ``modules`` vacío (lo impone
    :func:`app.generation.rag.grounding.is_coherent_estimate`).
    """

    total_engineer_days: int | None = None
    modules: list[EstimationModule] = Field(default_factory=list)
    duration_weeks: int | None = None
    sources: list[Citation] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    confidence: Confidence
    reasoning: str = Field(description="Cómo se derivó la estimación de las fuentes.")
    insufficient_context_explanation: str | None = None


# ---- Modelos HTTP de los routers S09 ----------------------------------------
# Nombrados `RetrievalQuery`/`TranscriptEstimateRequest` (no `SearchRequest`) para
# no colisionar con el contrato de la Sesión 8.


class RetrievalQuery(BaseModel):
    """Payload de ``POST /api/v1/retrieval/search`` (umbral + filtros)."""

    model_config = ConfigDict(extra="forbid")

    query_text: str = Field(min_length=10, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=30)
    distance_threshold: float = Field(default=0.6, ge=0.0, le=2.0)
    sectors: list[str] | None = None
    project_year_min: int | None = Field(default=None, ge=2010, le=2100)
    project_year_max: int | None = Field(default=None, ge=2010, le=2100)
    chunk_types: list[str] | None = None


class TranscriptEstimateRequest(BaseModel):
    """Payload de ``POST /api/v1/estimate/from-transcript``."""

    model_config = ConfigDict(extra="forbid")

    transcript: str = Field(min_length=100, max_length=50_000)
    idempotency_key: str | None = Field(default=None, max_length=128)


# ---- Modelos por etapa para el wizard --------------------------------------


class BriefRequest(BaseModel):
    """Payload de ``POST /api/v1/estimate/stages/brief``."""

    model_config = ConfigDict(extra="forbid")

    transcript: str = Field(min_length=100, max_length=50_000)


class BriefOutcome(BaseModel):
    """Salida de la comprensión: brief estructurado + texto canónico de búsqueda."""

    brief: ProjectBrief
    retrieval_text: str = Field(description="Texto alineado con el corpus que se embebe.")


class ContextAssemblyRequest(BaseModel):
    """Payload de ``POST /api/v1/estimate/stages/assemble``.

    ``max_context_tokens`` cae al valor del servidor (``RAG_MAX_CONTEXT_TOKENS``)
    si se omite; un valor pequeño permite demostrar el truncado por chunks.
    """

    model_config = ConfigDict(extra="forbid")

    chunks: list[RetrievedChunk]
    max_context_tokens: int | None = Field(default=None, ge=256, le=64_000)


class ContextAssemblyOutcome(BaseModel):
    """Salida de la augmentación: bloque ``<source>`` y qué sobrevivió."""

    context_block: str
    kept_chunks: list[RetrievedChunk]
    dropped_count: int = Field(ge=0, description="Chunks descartados por el presupuesto.")
    token_count: int = Field(ge=0, description="Tokens del bloque de contexto.")


class GroundedGenerationRequest(BaseModel):
    """Payload de ``POST /api/v1/estimate/stages/generate``.

    ``kept_chunks`` son los chunks con los que se construyó el contexto; se usan
    para validar citas (sin source ids inventados) tras la generación.
    """

    model_config = ConfigDict(extra="forbid")

    context_block: str = Field(min_length=1)
    brief: ProjectBrief
    kept_chunks: list[RetrievedChunk] = Field(default_factory=list)


class GroundedGenerationOutcome(BaseModel):
    """Salida de la generación: estimación + señales de grounding para el wizard."""

    estimate: GroundedEstimate
    fabricated_source_ids: list[int] = Field(
        default_factory=list,
        description="Source ids citados que no están en kept_chunks (vacío = limpio).",
    )
    coherent: bool = Field(
        description="False cuando una estimación insuficiente aún lleva números."
    )
