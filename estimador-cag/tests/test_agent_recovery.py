"""Fase 2 del agente: recuperación de horas con un cliente guionizado (sin red)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from app.generation.agentic.estimation_agent.recovery_agent import (
    run_task_hours_recovery_agent,
)
from app.generation.agentic.estimation_agent.schemas import AgentTaskRef

TASKS = [
    AgentTaskRef(
        module="Integración",
        task="Mapeo de IDoc",
        description="SAP IDoc con reintentos",
    )
]


def _reasoning(text: str) -> Any:
    return SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text=text)])


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=json.dumps(arguments),
    )


def _turn(response_id: str, output: list[Any]) -> Any:
    return SimpleNamespace(id=response_id, output=output)


class FakeResponses:
    def __init__(self, turns: list[Any]) -> None:
        self.turns = list(turns)
        self.create_calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return self.turns.pop(0)


class FakeClient:
    def __init__(self, turns: list[Any]) -> None:
        self.responses = FakeResponses(turns)


async def _backend(_args: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": 7,
            "budget_id": "BUD-2024-007",
            "sector": "industrial",
            "chunk_type": "historical_task",
            "content_preview": "SAP IDoc mapping",
            "estimated_hours": 60,
            "distance": 0.09,
        }
    ]


def _consensus(neighbors: list[tuple[int, float]]) -> tuple[int, float, float]:
    assert neighbors == [(60, 0.09)]
    return 42, 0.8, 0.1


async def test_recovery_loop_derives_hours_from_tool_observations() -> None:
    client = FakeClient(
        [
            _turn(
                "resp-1",
                [
                    _reasoning("Reformulo la búsqueda del mapeo de IDoc."),
                    _call(
                        "call-1",
                        "search_budgets",
                        {"query": "SAP IDoc mapping", "filters": None},
                    ),
                ],
            ),
            _turn(
                "resp-2",
                [
                    _call(
                        "call-2",
                        "derive_task_hours",
                        {
                            "module": "Integración",
                            "task": "Mapeo de IDoc",
                            "neighbors": [
                                {"source_id": 7, "estimated_hours": 60, "distance": 0.09}
                            ],
                        },
                    ),
                ],
            ),
            _turn("resp-3", []),
        ]
    )

    run = await run_task_hours_recovery_agent(
        TASKS,
        client=client,
        model="gpt-5",
        backend=_backend,
        consensus_fn=_consensus,
    )

    assert len(run.recovered) == 1
    derivation = run.recovered[0]
    assert derivation.module == "Integración"
    assert derivation.task == "Mapeo de IDoc"
    assert derivation.estimated_hours == 42
    assert derivation.reliability == 0.8
    assert derivation.source_chunk_ids == [7]
    assert [step.tool for step in run.trace.steps] == ["search_budgets", "derive_task_hours"]
    assert run.iterations == 3
    assert run.stopped_reason == "completed"
    assert "tools" in client.responses.create_calls[0]


async def test_recovery_respects_max_iterations() -> None:
    call = _call("call-1", "search_budgets", {"query": "algo", "filters": None})
    client = FakeClient([_turn(f"resp-{index}", [call]) for index in range(5)])

    run = await run_task_hours_recovery_agent(
        TASKS,
        client=client,
        model="gpt-5",
        backend=_backend,
        consensus_fn=_consensus,
        max_iterations=2,
    )

    assert run.stopped_reason == "max_iterations"
    assert run.iterations == 2


async def test_recovery_without_flagged_tasks_does_not_call_the_model() -> None:
    client = FakeClient([])

    run = await run_task_hours_recovery_agent(
        [],
        client=client,
        model="gpt-5",
        backend=_backend,
        consensus_fn=_consensus,
    )

    assert run.recovered == []
    assert run.iterations == 0
    assert client.responses.create_calls == []


async def test_derive_without_consensus_becomes_an_error_observation() -> None:
    client = FakeClient(
        [
            _turn(
                "resp-1",
                [
                    _call(
                        "call-1",
                        "derive_task_hours",
                        {
                            "module": "Integración",
                            "task": "Mapeo de IDoc",
                            "neighbors": [
                                {"source_id": 7, "estimated_hours": 60, "distance": 0.09}
                            ],
                        },
                    )
                ],
            ),
            _turn("resp-2", []),
        ]
    )

    run = await run_task_hours_recovery_agent(
        TASKS,
        client=client,
        model="gpt-5",
        backend=_backend,
        consensus_fn=None,  # type: ignore[arg-type]
    )

    assert run.recovered == []
    assert "needs an injected consensus function" in run.trace.steps[0].observation
