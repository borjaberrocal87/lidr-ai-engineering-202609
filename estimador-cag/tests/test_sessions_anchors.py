"""Tests del detector de anclas (compresión de memoria)."""

from typing import Any

from app.sessions.compression.anchors import AnchorDetector
from app.sessions.models import Message


def _msg(text: str) -> Message:
    return Message(role="user", content=text)


def test_heuristic_detects_nda() -> None:
    match = AnchorDetector().detect(_msg("We signed the NDA with the client last week."))

    assert match.is_anchor is True
    assert "signed_contract" in match.matched_rules or "nda" in match.matched_rules


def test_heuristic_detects_frozen_scope_and_compliance() -> None:
    assert AnchorDetector().detect(_msg("The scope is frozen for phase 1.")).is_anchor
    assert AnchorDetector().detect(_msg("This must be GDPR compliant.")).is_anchor


def test_heuristic_ignores_plain_turn() -> None:
    match = AnchorDetector().detect(_msg("Can we add a dashboard for the sales team?"))

    assert match.is_anchor is False
    assert match.matched_rules == []


class _FakeWrapper:
    def __init__(self, classification: Any = None, error: Exception | None = None) -> None:
        self.classification = classification
        self.error = error
        self.calls = 0

    def complete_structured_messages(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.classification, {"model": "gpt-4o-mini"}


def test_llm_mode_uses_wrapper_classification() -> None:
    from app.sessions.compression.anchors import _AnchorClassification

    wrapper = _FakeWrapper(
        classification=_AnchorClassification(is_anchor=True, reason="locked budget")
    )
    detector = AnchorDetector(mode="llm", llm_wrapper=wrapper)

    match = detector.detect(_msg("Whatever the user said."))

    assert match.is_anchor is True
    assert wrapper.calls == 1
    assert match.matched_rules[0].startswith("llm:")


def test_llm_mode_falls_back_to_heuristic_on_error() -> None:
    detector = AnchorDetector(mode="llm", llm_wrapper=_FakeWrapper(error=RuntimeError("boom")))

    match = detector.detect(_msg("We signed the SOW."))

    assert match.is_anchor is True
    assert "signed_contract" in match.matched_rules


def test_llm_mode_without_wrapper_uses_heuristic() -> None:
    detector = AnchorDetector(mode="llm")

    assert detector.detect(_msg("The backlog is locked.")).is_anchor is True
