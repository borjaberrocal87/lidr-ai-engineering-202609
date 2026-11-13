"""Plan de consulta: expansión y descomposición (Sesión 10).

Dos técnicas DISTINTAS que no son intercambiables. La pregunta que las separa es:
*¿una cosa dicha de muchas formas, o muchas cosas dichas a la vez?*

* **Expansión (multi-query)** — N reformulaciones de la MISMA intención. Combate
  la lotería de la formulación (que el vocabulario de la consulta decida qué se
  recupera). Se fusiona por consenso (RRF): sube el chunk en el que coinciden
  varias variantes.
* **Descomposición** — sub-consultas independientes, una por TEMA. Combate el
  embedding promediado de una consulta multitema ("cerca de todo, cerca de
  nada"). Se fusiona por cobertura (round-robin): cada tema tiene representación.

Cuál aplicar (si alguna) es una heurística barata sobre longitud y estructura de
la consulta; la decisión se registra. El LLM usa salida estructurada sobre un
modelo pequeño y rápido, con instrucciones de correa corta: conserva los términos
de dominio exactos, no inventes requisitos, menos es mejor que fragmentar, y como
mucho 4 sub-consultas (el tope vive en el esquema Y en las instrucciones).

El wrapper LLM se inyecta: este módulo no importa el composition root y respeta
el contrato de capas de ``test_layering.py``.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import structlog
from pydantic import BaseModel, Field

from app.config import get_settings
from app.foundation.prompts.loader import render_rag_query_plan_prompt

log = structlog.get_logger()

# Etiquetas de técnica (también seleccionan la estrategia de fusión aguas abajo).
DIRECT = "direct"
EXPAND = "expand"
DECOMPOSE = "decompose"

# Umbrales de la heurística (deliberadamente simples, ajustables en el directo).
_SHORT_WORDS = 6
_LONG_WORDS = 25
_MULTITOPIC_CONNECTORS = 2


class SubQuery(BaseModel):
    """Una consulta generada y el tema que cubre (el tema ayuda a trazar cobertura)."""

    topic: str = Field(description="Etiqueta corta del ángulo o tema que cubre.")
    query: str = Field(description="Consulta de búsqueda autocontenida.")


class QueryTransformation(BaseModel):
    """Salida estructurada: 1-4 sub-consultas (tope en esquema y en prompt)."""

    subqueries: list[SubQuery] = Field(min_length=1, max_length=4)


class QueryPlan(BaseModel):
    """La técnica elegida y las sub-consultas que hay que ejecutar."""

    technique: str
    subqueries: list[SubQuery]


def choose_technique(query_text: str) -> str:
    """Elige técnica según la forma de la consulta (longitud + conectores)."""
    words = len(query_text.split())
    if words < _SHORT_WORDS:
        return DIRECT

    lowered = query_text.lower()
    connectors = (
        lowered.count(" and ")
        + query_text.count(",")
        + query_text.count(";")
        + lowered.count(" plus ")
        + lowered.count(" also ")
        + lowered.count(" y ")
        + lowered.count(" además ")
    )
    sentences = sum(query_text.count(mark) for mark in ".!?")
    if words >= _LONG_WORDS or connectors >= _MULTITOPIC_CONNECTORS or sentences >= 2:
        return DECOMPOSE
    return EXPAND


async def transform_query(
    query_text: str,
    *,
    enabled: bool = True,
    llm_wrapper: Any = None,
    model: str | None = None,
    max_subqueries: int = 4,
) -> QueryPlan:
    """Devuelve las sub-consultas a ejecutar y la técnica usada.

    Ante cualquier fallo (o si está desactivado) degrada a una única consulta
    directa: la recuperación nunca depende de que el transformador acierte.
    """
    technique = choose_technique(query_text) if enabled else DIRECT
    if technique == DIRECT or llm_wrapper is None:
        log.info("query_plan_decided", technique=DIRECT, subqueries=1)
        return QueryPlan(
            technique=DIRECT,
            subqueries=[SubQuery(topic="original", query=query_text)],
        )

    system_prompt, user_message = render_rag_query_plan_prompt(
        technique=technique,
        query_text=query_text,
    )
    try:
        result = await _generate(
            user_message=user_message,
            system_prompt=system_prompt,
            llm_wrapper=llm_wrapper,
            model=model,
        )
        subqueries = result.subqueries[:max_subqueries]
        if not subqueries:
            raise ValueError("el transformador no devolvió sub-consultas")
        log.info(
            "query_plan_decided",
            technique=technique,
            subqueries=len(subqueries),
            topics=[sq.topic for sq in subqueries],
        )
        return QueryPlan(technique=technique, subqueries=subqueries)
    except Exception as exc:
        log.warning(
            "query_plan_failed",
            technique=technique,
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return QueryPlan(
            technique=DIRECT,
            subqueries=[SubQuery(topic="original", query=query_text)],
        )


async def _generate(
    *,
    system_prompt: str,
    user_message: str,
    llm_wrapper: Any,
    model: str | None,
) -> QueryTransformation:
    settings = get_settings()
    transformation, _meta = await asyncio.to_thread(
        llm_wrapper.complete_structured,
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=QueryTransformation,
        model_override=model or settings.query_plan_model,
        max_tokens=1500,
    )
    return cast(QueryTransformation, transformation)
