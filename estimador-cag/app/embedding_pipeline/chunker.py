"""Chunker estructural para presupuestos JSON.

Estrategia: confiar en la estructura del documento. Un ``BudgetComponent`` se
convierte en exactamente un ``Chunk``. Sin overlap y sin splitting a tamaño fijo
de las descripciones largas: si una descripción es anormalmente grande, es un
dato que ``token_count`` surfacea y queremos discutir, no tapar.

A cada chunk se le antepone el contexto del presupuesto padre (la *cabecera de
contexto*): sin ella, un componente como "Authentication backend" perdería la
pista de a qué cliente y sector pertenece.
"""

from __future__ import annotations

import tiktoken

from app.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk

# Tokenizador contra el que contamos. Cargar el encoding es relativamente caro;
# se resuelve una vez a nivel de módulo y se reutiliza.
TOKENIZER_MODEL = "text-embedding-3-small"
_ENCODING = tiktoken.encoding_for_model(TOKENIZER_MODEL)

# Plantillas del texto embebible. La cabecera aporta el contexto del padre; el
# cuerpo, el detalle del componente. Se mantienen separadas para que el test (y
# la lectura) puedan razonar sobre cada mitad por su cuenta.
_HEADER_TEMPLATE = (
    "[Project: {project_summary}]\n"
    "[Client sector: {sector} | Year: {year} | Main tech: {main_technology}]"
)
_BODY_TEMPLATE = (
    "Component: {name}\n"
    "Description: {description}\n"
    "Tech stack: {tech_stack}\n"
    "Complexity: {complexity}\n"
    "Estimated hours: {estimated_hours}"
)


def count_tokens(text: str) -> int:
    """Tokens de ``text`` con el tokenizador del modelo del ejercicio."""
    return len(_ENCODING.encode(text))


class JSONStructuralChunker:
    """Convierte presupuestos en un chunk por componente."""

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        return [
            self._chunk_component(budget, component)
            for budget in budgets
            for component in budget.components
        ]

    def _chunk_component(self, budget: Budget, component: BudgetComponent) -> Chunk:
        text = self._render_text(budget, component)
        return Chunk(
            chunk_id=f"{budget.budget_id}::{component.component_id}",
            text=text,
            metadata=self._build_metadata(budget, component),
            token_count=count_tokens(text),
        )

    @staticmethod
    def _render_text(budget: Budget, component: BudgetComponent) -> str:
        """Cabecera de contexto del padre + detalle del componente; esto se embebe."""
        header = _HEADER_TEMPLATE.format(
            project_summary=budget.project_summary,
            sector=budget.client_metadata.sector.value,
            year=budget.year,
            main_technology=budget.main_technology,
        )
        body = _BODY_TEMPLATE.format(
            name=component.name,
            description=component.description,
            tech_stack=", ".join(component.tech_stack),
            complexity=component.complexity.value,
            estimated_hours=component.estimated_hours,
        )
        return f"{header}\n\n{body}"

    @staticmethod
    def _build_metadata(budget: Budget, component: BudgetComponent) -> dict[str, str | int]:
        """Campos filtrables que viajan con el chunk pero no se embeben."""
        return {
            "budget_id": budget.budget_id,
            "component_id": component.component_id,
            "client_sector": budget.client_metadata.sector.value,
            "main_technology": budget.main_technology,
            "year": budget.year,
            "complexity": component.complexity.value,
            "estimated_hours": component.estimated_hours,
        }
