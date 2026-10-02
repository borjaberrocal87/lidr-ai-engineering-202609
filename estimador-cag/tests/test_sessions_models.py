"""Tests unitarios de ConversationHistory, ProjectMetadata y SessionStore."""

from __future__ import annotations

import pytest

from app.sessions.models import ConversationHistory, ProjectMetadata, Session
from app.sessions.store import SessionNotFoundError, SessionStore


def test_history_sliding_window_drops_oldest_pairs() -> None:
    history = ConversationHistory(max_turns=2)
    history.append(user="t1u", assistant="t1a")
    history.append(user="t2u", assistant="t2a")
    history.append(user="t3u", assistant="t3a")

    messages = history.to_messages_list("SYSTEM")
    assert len(messages) == 5  # system + 2 turns * 2
    assert messages[0] == {"role": "system", "content": "SYSTEM"}
    assert messages[1] == {"role": "user", "content": "t2u"}
    assert messages[-1] == {"role": "assistant", "content": "t3a"}


def test_history_keeps_role_alternation() -> None:
    history = ConversationHistory(max_turns=3)
    for index in range(5):
        history.append(user=f"u{index}", assistant=f"a{index}")

    roles = [message.role for message in history.messages]
    assert roles == ["user", "assistant"] * 3


def test_project_metadata_is_empty_by_default() -> None:
    metadata = ProjectMetadata()
    assert metadata.is_empty()
    assert metadata.mentioned_technologies == []


def test_project_metadata_merge_overwrites_scalars_and_unions_techs() -> None:
    base = ProjectMetadata(
        project_name="Nimbus CRM",
        assumed_team_size=3,
        mentioned_technologies=["React", "Postgres"],
        agreed_scope="Fase 1",
    )
    update = ProjectMetadata(
        project_name="Nimbus CRM v2",
        assumed_team_size=None,
        mentioned_technologies=["postgres", "Redis"],
        agreed_scope=None,
    )

    merged = base.merge_with(update)

    assert merged.project_name == "Nimbus CRM v2"
    assert merged.assumed_team_size == 3
    assert merged.agreed_scope == "Fase 1"
    assert merged.mentioned_technologies == ["React", "Postgres", "Redis"]


def test_session_store_creates_unique_ids_and_respects_max_turns() -> None:
    store = SessionStore(max_turns=4)
    first = store.create()
    second = store.create()

    assert first.session_id != second.session_id
    assert first.history.max_turns == 4
    assert len(store) == 2


def test_session_store_get_or_404_raises_for_unknown() -> None:
    store = SessionStore()
    with pytest.raises(SessionNotFoundError):
        store.get_or_404("nope")


def test_session_round_trips_through_json() -> None:
    session = Session()
    session.history.append(user="hi", assistant="hello")
    session.metadata = ProjectMetadata(project_name="Nimbus")

    restored = Session.model_validate(session.model_dump(mode="json"))

    assert restored.metadata.project_name == "Nimbus"
    assert len(restored.history.messages) == 2
