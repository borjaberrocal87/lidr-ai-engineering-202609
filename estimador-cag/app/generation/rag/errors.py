"""Errores del pipeline RAG end-to-end (Sesión 9).

La capa HTTP traduce ``RagPipelineError`` a un 502 genérico: el detalle interno
(qué etapa falló y por qué) se queda en los logs estructurados, no en la
respuesta al cliente. Los tests, en cambio, pueden distinguir la subclase para
verificar que el fallo ocurrió donde tocaba.
"""

from __future__ import annotations


class RagPipelineError(Exception):
    """Error base de cualquier etapa del pipeline RAG."""

    stage = "rag"


class BriefExtractionError(RagPipelineError):
    """No se pudo destilar un ``ProjectBrief`` de la transcripción."""

    stage = "brief"


class RagRetrievalError(RagPipelineError):
    """Falló el embedding de la consulta o la búsqueda filtrada."""

    stage = "retrieval"


class ContextAssemblyError(RagPipelineError):
    """No se pudo construir el bloque de contexto dentro del presupuesto."""

    stage = "assembly"


class RagGenerationError(RagPipelineError):
    """El LLM no devolvió una estimación grounded válida."""

    stage = "generation"


class IncoherentEstimateError(RagPipelineError):
    """La estimación incumple la regla de insuficiencia o de citas."""

    stage = "grounding"
