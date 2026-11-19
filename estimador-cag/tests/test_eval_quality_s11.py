"""Piezas deterministas del arnés de calidad de generación (Sesión 11)."""

from __future__ import annotations

from scripts.eval_quality_s11 import (
    ALL_CONFIG_IDS,
    CONFIGS,
    _gate_failures,
    summarize,
)


def test_configs_cover_the_live_toggle_combinations() -> None:
    assert ALL_CONFIG_IDS == ("baseline", "gate", "augment", "full")
    assert CONFIGS["full"].gate is True and CONFIGS["full"].augment is True
    assert CONFIGS["baseline"].gate is False and CONFIGS["baseline"].augment is False


def test_summarize_aggregates_citations_and_semantics() -> None:
    samples = [
        {
            "context_tokens": 100,
            "citation_report": {"total_lines": 4, "grounded_lines": 3, "dangling_citations": []},
            "hallucination_report": {
                "total_lines": 4,
                "grounded_lines": 3,
                "degraded_lines": 1,
            },
        },
        {
            "context_tokens": 300,
            "citation_report": {"total_lines": 2, "grounded_lines": 2, "dangling_citations": []},
            "hallucination_report": {
                "total_lines": 2,
                "grounded_lines": 2,
                "degraded_lines": 0,
            },
        },
    ]

    summary = summarize(samples)

    assert summary["samples"] == 2
    assert summary["lines"] == 6
    assert summary["citation_grounding_rate"] == 5 / 6
    assert summary["semantic_lines"] == 6
    assert summary["semantic_grounding_rate"] == 5 / 6
    assert summary["degraded_rate"] == 1 / 6
    assert summary["avg_context_tokens"] == 200


def test_summarize_without_gate_leaves_semantics_unmeasured() -> None:
    summary = summarize(
        [
            {
                "context_tokens": 10,
                "citation_report": {"total_lines": 1, "grounded_lines": 1},
                "hallucination_report": None,
            }
        ]
    )

    assert summary["semantic_lines"] == 0
    assert summary["semantic_grounding_rate"] == 0.0


def test_gate_flags_dangling_citations_and_low_grounding() -> None:
    failures = _gate_failures(
        {
            "dangling_citations": 2,
            "citation_grounding_rate": 0.5,
            "semantic_lines": 0,
            "degraded_rate": 0.0,
        }
    )

    assert any("colgantes" in failure for failure in failures)
    assert any("citacion" in failure for failure in failures)


def test_gate_flags_excessive_degradation_only_when_measured() -> None:
    degraded = _gate_failures(
        {
            "dangling_citations": 0,
            "citation_grounding_rate": 1.0,
            "semantic_lines": 4,
            "degraded_rate": 0.5,
        }
    )
    unmeasured = _gate_failures(
        {
            "dangling_citations": 0,
            "citation_grounding_rate": 1.0,
            "semantic_lines": 0,
            "degraded_rate": 0.5,
        }
    )

    assert any("degradadas" in failure for failure in degraded)
    assert unmeasured == []


async def test_synthesis_demo_surfaces_the_fixture_range() -> None:
    from scripts.eval_quality_s11 import synthesis_demo

    demo = await synthesis_demo()

    assert demo["neighbour_hours"] == [40, 90]
    assert demo["range"] == [40, 90]
    assert demo["reason"]
