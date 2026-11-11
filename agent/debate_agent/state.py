"""Debate-specific fields layered onto the common chat workflow state."""

from typing import TypedDict

from agent.schemas.state import BaseState


class DebateState(BaseState, total=False):
    practice_topic: str
    practice_domain: str
    practice_user_position: str
    practice_jurisdiction: str
    practice_ready: bool
    practice_started: bool
    practice_request_coaching: bool
    practice_retrieved_contexts: list[dict[str, object]]
    practice_last_argument: str
