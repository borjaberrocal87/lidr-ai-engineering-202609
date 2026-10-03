"""Almacén de sesiones en memoria, indexado por `session_id`.

La clase existe para aislar el punto de sustitución (Redis, Postgres…) sin
tocar el router ni el servicio. No es thread-safe: `uvicorn --workers=1` es el
escenario soportado; con más de un worker cada uno tendría su propia copia y se
rompería la garantía conversacional.
"""

from __future__ import annotations

from app.sessions.models import ConversationHistory, Session


class SessionNotFoundError(KeyError):
    """Se lanza cuando el `session_id` no existe."""


class SessionStore:
    def __init__(self, *, max_turns: int = 6) -> None:
        self._sessions: dict[str, Session] = {}
        self._max_turns = max_turns

    def create(self) -> Session:
        session = Session(history=ConversationHistory(max_turns=self._max_turns))
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_404(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise SessionNotFoundError(session_id) from exc

    def __len__(self) -> int:
        return len(self._sessions)
