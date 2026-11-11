"""Nullable structured fields used by the debate supervisor."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DebateSupervisorOutput(BaseModel):
    """All fields are optional so provider nulls or omissions remain valid."""

    model_config = ConfigDict(extra="ignore")

    domain: Literal["rent", "consumer", "wages", "uncertain", "general"] | None = Field(
        default=None,
        description="Supported vertical, uncertain, or outside current supported areas.",
    )
    topic: str | None = Field(default=None, description="The practice issue, if clear.")
    user_position: str | None = Field(
        default=None, description="The position the user wants to argue, if stated."
    )
    jurisdiction: str | None = Field(
        default=None, description="Jurisdiction stated or confirmed in message history."
    )
    ready: bool | None = Field(
        default=None, description="True only after clarification or confirmation is complete."
    )
    clarification_question: str | None = Field(
        default=None, description="One concise clarification question when needed."
    )
    request_coaching: bool | None = Field(
        default=None, description="True when coaching is requested during an active practice."
    )
