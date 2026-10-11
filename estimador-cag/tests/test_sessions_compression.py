"""Tests de la política de compresión y del resumidor acumulativo."""

from typing import Any

from app.sessions.compression.anchors import AnchorDetector
from app.sessions.compression.policy import CompressionPolicy
from app.sessions.compression.summarizer import CumulativeSummarizer, SummaryEnvelope
from app.sessions.models import ConversationHistory, Message


class _StubSummarizer:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def summarize(self, *, previous_summary: str | None, evicted: list[Message]) -> str:
        self.calls.append(evicted)
        return f"RESUMEN({len(evicted)} msgs)"


def _policy() -> tuple[CompressionPolicy, _StubSummarizer]:
    summarizer = _StubSummarizer()
    policy = CompressionPolicy(
        anchor_detector=AnchorDetector(mode="heuristic"),
        summarizer=summarizer,  # type: ignore[arg-type]
    )
    return policy, summarizer


def test_policy_is_noop_when_under_cap() -> None:
    history = ConversationHistory(max_turns=3)
    history.append(user="u1", assistant="a1")
    history.append(user="u2", assistant="a2")

    policy, _summarizer = _policy()
    policy.apply(history)

    assert len(history.messages) == 4
    assert history.summary is None
    assert _summarizer.calls == []


def test_policy_evicts_pairs_and_summarizes() -> None:
    history = ConversationHistory(max_turns=3)
    for index in range(5):
        history.append(user=f"u{index}", assistant=f"a{index}")

    policy, summarizer = _policy()
    policy.apply(history)

    assert len(history.messages) <= 3 * 2
    assert history.summary == "RESUMEN(4 msgs)"
    assert history.anchors == []
    assert len(summarizer.calls) == 1


def test_policy_promotes_anchor_turn_and_keeps_it_verbatim() -> None:
    history = ConversationHistory(max_turns=2)
    history.append(user="The scope is frozen for phase 1.", assistant="Entendido.")
    for index in range(3):
        history.append(user=f"u{index}", assistant=f"a{index}")

    policy, _summarizer = _policy()
    policy.apply(history)

    assert len(history.anchors) == 2
    assert history.anchors[0].content == "The scope is frozen for phase 1."
    assert len(history.messages) <= 2 * 2
    assert history.summary is not None


def test_policy_is_idempotent() -> None:
    history = ConversationHistory(max_turns=3)
    for index in range(5):
        history.append(user=f"u{index}", assistant=f"a{index}")

    policy, summarizer = _policy()
    policy.apply(history)
    first_summary = history.summary
    calls_after_first = len(summarizer.calls)

    policy.apply(history)

    assert history.summary == first_summary
    assert len(summarizer.calls) == calls_after_first


class _FakeWrapper:
    def __init__(self, envelope: Any = None, error: Exception | None = None) -> None:
        self.envelope = envelope or SummaryEnvelope(summary="Resumen rodante.")
        self.error = error
        self.calls = 0

    def complete_structured_messages(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.envelope, {"model": "gpt-4o-mini"}


def test_summarizer_returns_envelope_summary() -> None:
    wrapper = _FakeWrapper(envelope=SummaryEnvelope(summary="El cliente pidió Stripe."))
    summarizer = CumulativeSummarizer(llm_wrapper=wrapper, model="gpt-4o-mini")

    result = summarizer.summarize(
        previous_summary="Antes: React.",
        evicted=[Message(role="user", content="Añade facturación Stripe.")],
    )

    assert result == "El cliente pidió Stripe."
    assert wrapper.calls == 1


def test_summarizer_keeps_previous_on_error() -> None:
    wrapper = _FakeWrapper(error=RuntimeError("boom"))
    summarizer = CumulativeSummarizer(llm_wrapper=wrapper, model="gpt-4o-mini")

    result = summarizer.summarize(
        previous_summary="Resumen previo intacto.",
        evicted=[Message(role="user", content="algo")],
    )

    assert result == "Resumen previo intacto."


def test_summarizer_without_evicted_returns_previous() -> None:
    summarizer = CumulativeSummarizer(llm_wrapper=_FakeWrapper(), model="gpt-4o-mini")

    assert summarizer.summarize(previous_summary=None, evicted=[]) == ""
    assert summarizer.summarize(previous_summary="X", evicted=[]) == "X"
