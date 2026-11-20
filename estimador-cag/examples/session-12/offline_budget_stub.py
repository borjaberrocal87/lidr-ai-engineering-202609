"""Offline budget-search backend for debugging the agent loop (Session 12).

This is NOT the delivery path: the real agent runs ``search_budgets`` against the
hybrid retrieval pipeline of Sessions 9-10. This stub exists only to exercise the
loop mechanics (argument parsing, ``call_id`` round-trip, stop conditions) without
Docker, Postgres or the ingested corpus.

It is our own fixture, not a copy of the course stub: it mirrors the historical
components that live in ``data/budgets_sample.json`` so the numbers it returns are
in the same ballpark as a real run.

Usage (from ``scripts/run_agent_s12.py --stub``)::

    from offline_budget_stub import search_budgets_stub
    items = search_budgets_stub("SAP ERP integration with IDoc mapping", None)
"""

from __future__ import annotations

import re
from typing import Any

_CATALOG: list[dict[str, Any]] = [
    {
        "id": 101,
        "budget_id": "BUD-2024-001",
        "component_type": "authentication backend",
        "sector": "finance",
        "content_preview": (
            "OAuth 2.0 authentication backend with authorization code and refresh "
            "token flows, JWT session management, multi-tenant token isolation and "
            "per-client rate limiting."
        ),
        "estimated_hours": 120,
    },
    {
        "id": 102,
        "budget_id": "BUD-2024-003",
        "component_type": "payments backend",
        "sector": "finance",
        "content_preview": (
            "Payment gateway core processing real-time transactions with a "
            "double-entry ledger, settlement reconciliation and merchant dashboard."
        ),
        "estimated_hours": 220,
    },
    {
        "id": 103,
        "budget_id": "BUD-2024-007",
        "component_type": "business backend",
        "sector": "logistics",
        "content_preview": (
            "Logistics business backend: order management, delivery slot booking, "
            "route assignment, shipment tracking and tariff calculation behind a "
            "REST API."
        ),
        "estimated_hours": 210,
    },
    {
        "id": 104,
        "budget_id": "BUD-2024-007",
        "component_type": "route optimization",
        "sector": "logistics",
        "content_preview": (
            "Route optimization service computing delivery routes from orders, "
            "vehicle capacity and time windows, with real-time inventory sync."
        ),
        "estimated_hours": 150,
    },
    {
        "id": 105,
        "budget_id": "BUD-2024-014",
        "component_type": "erp integration",
        "sector": "industrial",
        "content_preview": (
            "ERP integration layer with middleware, master-data synchronization, "
            "IDoc and EDI mapping, retry handling and idempotent replay against "
            "SAP-style enterprise systems."
        ),
        "estimated_hours": 260,
    },
    {
        "id": 106,
        "budget_id": "BUD-2024-014",
        "component_type": "warehouse backend",
        "sector": "industrial",
        "content_preview": (
            "Warehouse management core with AGV coordination, yard and dock "
            "scheduling and operations KPI reporting."
        ),
        "estimated_hours": 240,
    },
    {
        "id": 107,
        "budget_id": "BUD-2024-017",
        "component_type": "mobile app",
        "sector": "retail",
        "content_preview": (
            "Cross-platform mobile app for field staff: daily task list, delivery "
            "confirmation with signature and photo, offline-first storage and "
            "background synchronization when connectivity returns."
        ),
        "estimated_hours": 190,
    },
    {
        "id": 108,
        "budget_id": "BUD-2024-010",
        "component_type": "analytics dashboard",
        "sector": "healthcare",
        "content_preview": (
            "Analytics dashboard with KPI cards, trend charts, filters by region "
            "and client, and threshold alerting for out-of-range metrics."
        ),
        "estimated_hours": 140,
    },
    {
        "id": 109,
        "budget_id": "BUD-2024-005",
        "component_type": "web storefront",
        "sector": "retail",
        "content_preview": (
            "Headless storefront with product catalog API, cart and checkout "
            "service, personalized recommendations and a React PWA front end."
        ),
        "estimated_hours": 150,
    },
    {
        "id": 110,
        "budget_id": "BUD-2024-009",
        "component_type": "healthcare integration",
        "sector": "healthcare",
        "content_preview": (
            "FHIR integration layer for clinical systems: patient records, "
            "appointment scheduling, telemedicine video sessions and consent with "
            "audit logging."
        ),
        "estimated_hours": 190,
    },
    {
        "id": 111,
        "budget_id": "BUD-2024-013",
        "component_type": "iot telemetry",
        "sector": "industrial",
        "content_preview": (
            "Sensor telemetry ingestion pipeline with predictive maintenance "
            "models, real-time equipment monitoring and SCADA dashboard integration."
        ),
        "estimated_hours": 170,
    },
    {
        "id": 112,
        "budget_id": "BUD-2024-015",
        "component_type": "data pipeline",
        "sector": "energy",
        "content_preview": (
            "High-volume data pipeline for smart meters with demand-response "
            "engine, load forecasting service and operational reporting."
        ),
        "estimated_hours": 180,
    },
]

_STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "una",
    "con",
    "para",
    "del",
    "las",
    "los",
    "por",
    "que",
}


def _tokens(text: str) -> set[str]:
    """Lowercase word tokens, dropping stopwords and 1-2 character noise."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def search_budgets_stub(
    query: str,
    filters: dict[str, Any] | None = None,
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Rank the canned catalog by token overlap with ``query``.

    Returns the same item shape the real backend produces, so it is a drop-in
    ``BudgetSearchBackend`` payload. ``distance`` is a crude ``1 - overlap`` so the
    agent still sees a monotonic relevance signal.
    """
    wanted_sectors = None
    if filters:
        sectors = filters.get("sectors")
        wanted_sectors = set(sectors) if sectors else None

    query_tokens = _tokens(query)
    if not query_tokens:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for item in _CATALOG:
        if wanted_sectors and item["sector"] not in wanted_sectors:
            continue
        document = _tokens(f"{item['component_type']} {item['content_preview']}")
        overlap = len(query_tokens & document)
        if overlap == 0:
            continue
        score = overlap / len(query_tokens)
        scored.append((score, {**item, "distance": round(1 - score, 4)}))

    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [item for _, item in scored[:limit]]
