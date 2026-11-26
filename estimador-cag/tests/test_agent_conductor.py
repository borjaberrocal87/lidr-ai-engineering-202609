"""Fase 2 híbrida: consenso determinista + recuperación solo de lo no groundado."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from app.domain.agent_estimation import agent_estimate_task_hours
from app.generation.rag.schemas import (
    TaskHoursEstimate,
    TaskHoursModuleInput,
    TaskHoursResult,
    TaskHoursTaskInput,
)

MODULES = [
    TaskHoursModuleInput(
        name="Auth",
        tasks=[TaskHoursTaskInput(name="Login", description="OAuth")],
    ),
    TaskHoursModuleInput(
        name="ERP",
        tasks=[TaskHoursTaskInput(name="IDoc", description="SAP")],
    ),
    TaskHoursModuleInput(
        name="Reporting",
        tasks=[TaskHoursTaskInput(name="Dashboards", description="BI")],
    ),
]

BASE = TaskHoursResult(
    tasks=[
        TaskHoursEstimate(
            module="Auth", task="Login", estimated_hours=40, reliability=0.9, has_match=True
        ),
        TaskHoursEstimate(module="ERP", task="IDoc", has_match=False),
        TaskHoursEstimate(
            module="Reporting",
            task="Dashboards",
            estimated_hours=10,
            reliability=0.2,
            has_match=True,
        ),
    ]
)


def _call(call_id: str, module: str, task: str) -> Any:
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name="derive_task_hours",
        arguments=json.dumps(
            {
                "module": module,
                "task": task,
                "neighbors": [{"source_id": 11, "estimated_hours": 30, "distance": 0.1}],
            }
        ),
    )


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


def _consensus(_neighbors: list[tuple[int, float]]) -> tuple[int, float, float]:
    return 55, 0.7, 0.05


async def _base() -> TaskHoursResult:
    return BASE


async def _backend(_args: Any) -> list[dict[str, Any]]:
    return []


def _turns_for_two_flagged() -> list[Any]:
    return [
        SimpleNamespace(id="resp-1", output=[_call("c1", "ERP", "IDoc")]),
        SimpleNamespace(id="resp-2", output=[_call("c2", "Reporting", "Dashboards")]),
        SimpleNamespace(id="resp-3", output=[]),
    ]


async def test_hybrid_only_sends_flagged_tasks_to_the_agent_and_merges() -> None:
    client = FakeClient(_turns_for_two_flagged())

    result = await agent_estimate_task_hours(
        MODULES,
        client=client,
        backend=_backend,
        consensus_fn=_consensus,
        base_estimator=_base,
        model="gpt-5",
        reasoning_effort="medium",
        max_iterations=5,
        low_reliability=0.35,
    )

    by_key = {(estimate.module, estimate.task): estimate for estimate in result.tasks}
    # La tarea fiable no se toca.
    assert by_key[("Auth", "Login")].estimated_hours == 40
    assert by_key[("Auth", "Login")].reliability == 0.9
    # La no groundada se recupera con la matemática inyectada.
    assert by_key[("ERP", "IDoc")].estimated_hours == 55
    assert by_key[("ERP", "IDoc")].has_match is True
    assert by_key[("ERP", "IDoc")].sources[0].chunk_id == "11"
    # La de fiabilidad baja también vuelve al agente.
    assert by_key[("Reporting", "Dashboards")].estimated_hours == 55
    assert result.agent_trace is not None
    assert len(result.agent_trace.steps) == 2
    # El mensaje inicial del agente solo lista las dos marcadas.
    first_input = client.responses.create_calls[0]["input"][0]["content"]
    assert "ERP | IDoc" in first_input
    assert "Reporting | Dashboards" in first_input
    assert "Auth | Login" not in first_input


async def test_hybrid_skips_the_agent_when_everything_is_grounded() -> None:
    grounded = TaskHoursResult(
        tasks=[
            TaskHoursEstimate(
                module="Auth",
                task="Login",
                estimated_hours=40,
                reliability=0.9,
                has_match=True,
            )
        ]
    )

    async def base() -> TaskHoursResult:
        return grounded

    client = FakeClient([])
    result = await agent_estimate_task_hours(
        [TaskHoursModuleInput(name="Auth", tasks=[TaskHoursTaskInput(name="Login")])],
        client=client,
        backend=_backend,
        consensus_fn=_consensus,
        base_estimator=base,
        model="gpt-5",
        reasoning_effort="medium",
        max_iterations=5,
    )

    assert result is grounded
    assert result.agent_trace is None
    assert client.responses.create_calls == []
