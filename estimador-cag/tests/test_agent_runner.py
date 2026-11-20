"""El bucle del agente con un cliente Responses guionizado (sin red)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from app.generation.agentic.estimation_agent.runner import run_estimation_agent
from app.generation.agentic.estimation_agent.schemas import (
    AgentComponent,
    AgentEstimateReport,
    SearchBudgetsArgs,
)

REPORT = AgentEstimateReport(
    components=[
        AgentComponent(
            name="Auth",
            estimated_hours=138.0,
            source_chunk_ids=[1],
            rationale="Grounded in an analogous authentication backend.",
        )
    ],
    total_hours=138.0,
    assumptions=[],
    confidence="high",
)


def _reasoning(text: str) -> Any:
    return SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text=text)])


def _call(call_id: str, name: str, arguments: dict[str, Any] | str) -> Any:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments=raw)


def _turn(response_id: str, output: list[Any]) -> Any:
    return SimpleNamespace(id=response_id, output=output)


class FakeResponses:
    def __init__(self, turns: list[Any], report: AgentEstimateReport | None = REPORT) -> None:
        self.turns = list(turns)
        self.report = report
        self.create_calls: list[dict[str, Any]] = []
        self.parse_calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return self.turns.pop(0)

    async def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.report)


class FakeClient:
    def __init__(self, turns: list[Any], report: AgentEstimateReport | None = REPORT) -> None:
        self.responses = FakeResponses(turns, report)


async def _backend(_args: SearchBudgetsArgs) -> list[dict[str, Any]]:
    return [
        {
            "id": 1,
            "budget_id": "BUD-2024-001",
            "sector": "finance",
            "chunk_type": "budget_component",
            "content_preview": "OAuth backend",
            "estimated_hours": 120,
            "distance": 0.1,
        }
    ]


async def test_loop_runs_two_tools_then_returns_the_report() -> None:
    client = FakeClient(
        [
            _turn(
                "resp-1",
                [
                    _reasoning("Four components; start with auth."),
                    _call(
                        "call-1",
                        "search_budgets",
                        {"query": "OAuth backend", "filters": None},
                    ),
                ],
            ),
            _turn(
                "resp-2",
                [
                    _call(
                        "call-2",
                        "calculate_estimate",
                        {"components": [{"name": "Auth", "reference_amounts": [120.0]}]},
                    )
                ],
            ),
            _turn("resp-3", []),
        ]
    )

    result = await run_estimation_agent(
        "transcript",
        client=client,
        model="gpt-5",
        reasoning_effort="medium",
        max_iterations=5,
        backend=_backend,
    )

    assert result.stopped_reason == "completed"
    assert result.report is not None and result.report.total_hours == 138.0
    assert [step.tool for step in result.trace.steps] == ["search_budgets", "calculate_estimate"]
    assert result.trace.steps[0].reasoning_summary == "Four components; start with auth."
    assert "1 historical items" in result.trace.steps[0].observation

    # El segundo create reenvía el output con el call_id de la call.
    follow_up = client.responses.create_calls[1]
    assert follow_up["previous_response_id"] == "resp-1"
    assert follow_up["input"][0]["type"] == "function_call_output"
    assert follow_up["input"][0]["call_id"] == "call-1"

    # El parseo final se encadena al último turno.
    assert client.responses.parse_calls[0]["previous_response_id"] == "resp-3"
    assert result.trace.iterations == result.iterations == 4


async def test_invalid_json_arguments_become_a_self_correctable_observation() -> None:
    client = FakeClient(
        [
            _turn("resp-1", [_call("call-1", "search_budgets", "{not-json")]),
            _turn("resp-2", []),
        ]
    )

    result = await run_estimation_agent(
        "transcript",
        client=client,
        model="gpt-5",
        max_iterations=5,
        backend=_backend,
    )

    assert result.stopped_reason == "completed"
    assert "not valid JSON" in result.trace.steps[0].observation
    output = client.responses.create_calls[1]["input"][0]["output"]
    assert "not valid JSON" in output


async def test_parallel_calls_in_one_turn_share_a_single_reasoning_summary() -> None:
    client = FakeClient(
        [
            _turn(
                "resp-1",
                [
                    _reasoning("Search two components at once."),
                    _call("call-1", "search_budgets", {"query": "auth", "filters": None}),
                    _call("call-2", "search_budgets", {"query": "mobile", "filters": None}),
                ],
            ),
            _turn("resp-2", []),
        ]
    )

    result = await run_estimation_agent(
        "transcript",
        client=client,
        model="gpt-5",
        max_iterations=5,
        backend=_backend,
    )

    assert len(result.trace.steps) == 2
    assert result.trace.steps[0].reasoning_summary == "Search two components at once."
    assert "parallel tool call" in (result.trace.steps[1].reasoning_summary or "")
    sent = client.responses.create_calls[1]["input"]
    assert [item["call_id"] for item in sent] == ["call-1", "call-2"]


async def test_a_tool_error_is_returned_to_the_model_without_breaking_the_loop() -> None:
    client = FakeClient(
        [
            _turn("resp-1", [_call("call-1", "calculate_estimate", {"components": "nope"})]),
            _turn("resp-2", []),
        ]
    )

    result = await run_estimation_agent(
        "transcript",
        client=client,
        model="gpt-5",
        max_iterations=5,
        backend=_backend,
    )

    assert result.stopped_reason == "completed"
    assert "ValidationError" in result.trace.steps[0].observation


async def test_the_max_iterations_safeguard_stops_the_loop_without_a_report() -> None:
    turns = [
        _turn(f"resp-{index}", [_call(f"call-{index}", "search_budgets", {"query": "x"})])
        for index in range(1, 6)
    ]
    client = FakeClient(turns)

    result = await run_estimation_agent(
        "transcript",
        client=client,
        model="gpt-5",
        max_iterations=2,
        backend=_backend,
    )

    assert result.stopped_reason == "max_iterations"
    assert result.report is None
    assert result.iterations == 2
    assert client.responses.parse_calls == []
