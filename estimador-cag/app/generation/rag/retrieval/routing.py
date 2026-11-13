"""Enrutado de fuentes en cascada de coste creciente (Sesión 10).

El corpus es heterogéneo; buscar todas las fuentes en cada consulta deja que el
tipo dominante inunde el top-k. El enrutador decide QUÉ fuentes buscar, en este
orden: la señal más barata que basta gana.

0. **Explícito** — el llamante nombró las fuentes en el contrato. El mejor
   enrutador es el que no enruta: si el cliente lo sabe, se confía y no se
   clasifica nada.
1. **Reglas** — patrones de vocabulario del catálogo (``sources.match_rules``).
   Una consulta que dice "presupuesto", "coste" u "horas" quiere presupuestos;
   no hace falta un LLM.
2. **Clasificador** — un LLM con salida estructurada emite 1-3 destinos y una
   razón de una frase. La ambigüedad se modela como VARIOS destinos, nunca como
   un umbral de confianza.
3. **Fallback** — buscar todas las fuentes en paralelo. Degradación elegante:
   nunca peor que el mundo de un solo índice del que venimos.

Cada decisión se registra con su nivel y su razón (atribución, auditoría y
depuración). Es un precursor acotado del relevo agéntico: clasificación cerrada,
sin razonamiento abierto y sin herramientas.

El wrapper LLM se inyecta: este módulo no importa el composition root.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import structlog
from pydantic import BaseModel, Field

from app.config import get_settings
from app.foundation.prompts.loader import render_rag_collection_router_prompt
from app.generation.rag.retrieval.sources import (
    Collection,
    all_collections,
    collections,
    match_rules,
)

log = structlog.get_logger()

EXPLICIT = "explicit"
RULES = "rules"
CLASSIFIER = "classifier"
FALLBACK = "fallback"


class RouteClassification(BaseModel):
    """Salida estructurada del clasificador LLM (destinos cerrados, 1-3)."""

    targets: list[Collection] = Field(
        min_length=1,
        max_length=3,
        description="Fuentes que hay que buscar. Una por tipo de fuente realmente "
        "relevante; modela la ambigüedad como varios destinos, no como un quizá.",
    )
    reason: str = Field(description="Una frase que justifica la decisión.")


class RoutingDecision(BaseModel):
    """Dónde se enruta la consulta y cómo se decidió."""

    targets: list[Collection]
    level: str
    reason: str


async def route(
    query_text: str,
    *,
    explicit: list[Collection] | None = None,
    rules_enabled: bool = True,
    classifier_enabled: bool = True,
    max_targets: int = 3,
    model: str | None = None,
    llm_wrapper: Any = None,
) -> RoutingDecision:
    """Resuelve las fuentes destino por la cascada descrita arriba."""
    # Nivel 0 — contrato explícito.
    if explicit:
        targets = list(dict.fromkeys(explicit))[:max_targets]
        decision = RoutingDecision(
            targets=targets,
            level=EXPLICIT,
            reason="El llamante especificó las fuentes.",
        )
        _log_decision(decision)
        return decision

    # Nivel 1 — reglas deterministas del catálogo.
    if rules_enabled:
        hits = match_rules(query_text)
        if hits:
            targets = hits[:max_targets]
            reason = f"Vocabulario compatible con: {', '.join(t.value for t in targets)}."
            decision = RoutingDecision(targets=targets, level=RULES, reason=reason)
            _log_decision(decision)
            return decision

    # Nivel 2 — clasificador LLM con salida estructurada.
    if classifier_enabled and llm_wrapper is not None:
        try:
            classification = await _classify(query_text, llm_wrapper=llm_wrapper, model=model)
            targets = list(dict.fromkeys(classification.targets))[:max_targets]
            decision = RoutingDecision(
                targets=targets,
                level=CLASSIFIER,
                reason=classification.reason,
            )
            _log_decision(decision)
            return decision
        except Exception as exc:
            log.warning(
                "routing_classifier_failed",
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )

    # Nivel 3 — fallback: buscar todas las fuentes.
    decision = RoutingDecision(
        targets=list(all_collections()),
        level=FALLBACK,
        reason="Sin señal de reglas ni del clasificador; se buscan todas las fuentes.",
    )
    _log_decision(decision)
    return decision


def _log_decision(decision: RoutingDecision) -> None:
    log.info(
        "routing_decided",
        level=decision.level,
        targets=[t.value for t in decision.targets],
        reason=decision.reason,
    )


async def _classify(
    query_text: str,
    *,
    llm_wrapper: Any,
    model: str | None,
) -> RouteClassification:
    """Ejecuta el clasificador LLM de fuentes (salida estructurada)."""
    settings = get_settings()
    catalog = [
        {
            "name": spec.collection.value,
            "label": spec.label,
            "description": spec.description,
        }
        for spec in collections().values()
    ]
    system_prompt, user_message = render_rag_collection_router_prompt(
        query_text=query_text,
        collections=catalog,
    )
    classification, _meta = await asyncio.to_thread(
        llm_wrapper.complete_structured,
        system_prompt=system_prompt,
        user_message=user_message,
        response_model=RouteClassification,
        model_override=model or settings.router_model,
        max_tokens=1000,
    )
    return cast(RouteClassification, classification)
