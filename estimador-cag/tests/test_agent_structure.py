"""Fase 1 del agente: parse de la estructura y mapeo al contrato del wizard."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.domain.agent_estimation import agent_propose_structure
from app.generation.agentic.estimation_agent.schemas import (
    AgentModule,
    AgentStructure,
    AgentTask,
)
from app.generation.agentic.estimation_agent.structure_agent import run_structure_agent
from app.generation.rag.schemas import ProjectBrief

STRUCTURE = AgentStructure(
    modules=[
        AgentModule(
            name="Backend de pedidos",
            description="API y persistencia",
            tasks=[
                AgentTask(name="Modelo de pedidos", description="CRUD y estados"),
                AgentTask(name="Pasarela de pago"),
            ],
        ),
        AgentModule(name="App móvil", tasks=[AgentTask(name="Login biométrico")]),
    ],
    assumptions=["El cliente aporta la pasarela de pago."],
    confidence="medium",
    reasoning="El brief separa backend y móvil.",
)


class FakeResponses:
    def __init__(self, structure: AgentStructure) -> None:
        self.structure = structure
        self.parse_calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.structure)


class FakeClient:
    def __init__(self, structure: AgentStructure) -> None:
        self.responses = FakeResponses(structure)


BRIEF = ProjectBrief(
    functionality="App de pedidos con backend y móvil",
    technologies=["FastAPI", "React Native"],
    sector="logistics",
)


async def test_structure_agent_returns_the_parsed_tree_and_a_trace() -> None:
    client = FakeClient(STRUCTURE)

    run = await run_structure_agent(
        "brief text",
        client=client,
        model="gpt-5",
        reasoning_effort="low",
    )

    assert [module.name for module in run.structure.modules] == [
        "Backend de pedidos",
        "App móvil",
    ]
    assert run.trace.model == "gpt-5"
    assert run.trace.iterations == 1
    assert run.trace.stopped_reason == "completed"
    assert client.responses.parse_calls[0]["text_format"] is AgentStructure
    assert "tools" not in client.responses.parse_calls[0]


async def test_conductor_maps_the_structure_to_the_wizard_contract() -> None:
    proposal = await agent_propose_structure(
        BRIEF,
        client=FakeClient(STRUCTURE),
        model="gpt-5",
        reasoning_effort="low",
    )

    estimate = proposal.estimate
    assert estimate.total_engineer_days is None
    assert estimate.confidence == "medium"
    assert [module.name for module in estimate.modules] == [
        "Backend de pedidos",
        "App móvil",
    ]
    first_task = estimate.modules[0].tasks[0]
    assert first_task.name == "Modelo de pedidos"
    assert first_task.grounded is False
    assert first_task.engineer_days is None
    assert first_task.sources == []
    assert estimate.assumptions[0].description == "El cliente aporta la pasarela de pago."
    assert proposal.trace.stopped_reason == "completed"
