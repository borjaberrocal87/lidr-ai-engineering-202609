"""Etapa de comprensión: transcripción → ProjectBrief."""

from __future__ import annotations

import pytest

from app.generation.rag.brief_extractor import compose_retrieval_text, extract_project_brief
from app.generation.rag.errors import BriefExtractionError
from app.generation.rag.schemas import ProjectBrief


class FakeBriefWrapper:
    def __init__(
        self,
        *,
        structured: ProjectBrief | None = None,
        structured_error: Exception | None = None,
        rewrite: str = "",
        rewrite_error: Exception | None = None,
    ) -> None:
        self.structured = structured or ProjectBrief(
            functionality="B2B payments marketplace",
            technologies=["python", "stripe"],
            sector="finance",
            country="ES",
            regulations=["PSD2"],
        )
        self.structured_error = structured_error
        self.rewrite = rewrite
        self.rewrite_error = rewrite_error
        self.structured_calls: list[dict] = []
        self.complete_calls: list[dict] = []

    def complete_structured(self, **kwargs):
        self.structured_calls.append(kwargs)
        if self.structured_error is not None:
            raise self.structured_error
        return self.structured, {"model": "gpt-5-mini"}

    def complete(self, **kwargs):
        self.complete_calls.append(kwargs)
        if self.rewrite_error is not None:
            raise self.rewrite_error
        return {"estimation": self.rewrite}


async def test_extract_project_brief_uses_structured_output(monkeypatch) -> None:
    wrapper = FakeBriefWrapper()
    monkeypatch.setattr("app.dependencies.get_llm_wrapper", lambda: wrapper)

    brief = await extract_project_brief("transcripción de la reunión")

    assert brief.functionality == "B2B payments marketplace"
    assert wrapper.structured_calls[0]["response_model"] is ProjectBrief
    assert wrapper.complete_calls == []


async def test_extract_project_brief_falls_back_to_rewrite(monkeypatch) -> None:
    wrapper = FakeBriefWrapper(
        structured_error=RuntimeError("structured output unsupported"),
        rewrite="Build a fintech payments dashboard",
    )
    monkeypatch.setattr("app.dependencies.get_llm_wrapper", lambda: wrapper)

    brief = await extract_project_brief("transcripción de la reunión")

    assert brief.functionality == "Build a fintech payments dashboard"
    assert brief.technologies == []
    assert wrapper.complete_calls


async def test_extract_project_brief_raises_when_both_paths_fail(monkeypatch) -> None:
    wrapper = FakeBriefWrapper(
        structured_error=RuntimeError("boom"),
        rewrite_error=RuntimeError("también boom"),
    )
    monkeypatch.setattr("app.dependencies.get_llm_wrapper", lambda: wrapper)

    with pytest.raises(BriefExtractionError):
        await extract_project_brief("transcripción de la reunión")


def test_compose_retrieval_text_joins_grounded_fields() -> None:
    brief = ProjectBrief(
        functionality="B2B payments marketplace",
        technologies=["Stripe Connect", "KYC"],
        sector="healthcare",
        country="Germany",
        regulations=["BaFin"],
    )

    text = compose_retrieval_text(brief)

    assert text == (
        "B2B payments marketplace with Stripe Connect, KYC "
        "for healthcare in Germany, BaFin-compliant"
    )
