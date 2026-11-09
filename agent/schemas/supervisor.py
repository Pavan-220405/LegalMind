"""Validated structured output produced by the supervisor model."""

from typing import Literal

from pydantic import BaseModel, Field


class SupervisorOutput(BaseModel):
    """Issue facts and routing decision extracted from the current turn."""

    issue_title: str = Field(description="Short title for the user's legal issue.")
    problem_statement: str = Field(
        description="A concise, neutral summary using only provided facts."
    )
    timeline_raw: str = Field(
        default="",
        description="The user's timeline wording, preserved as stated; empty if absent."
    )
    timeline_parsed: str = Field(
        default="",
        description=(
            "An explicit date or date range only when the user gave one; otherwise empty. "
            "Do not calculate dates from relative phrases."
        )
    )
    proofs_stated: list[str] = Field(
        default_factory=list,
        description="Evidence the user explicitly says they have or provided in text."
    )
    proofs_from_pdf: list[str] = Field(
        default_factory=list,
        description="Evidence identified in uploaded PDF text, kept distinct from user claims."
    )
    gender: Literal["male", "female", "unspecified"] = Field(
        default="unspecified",
        description="Use unspecified unless the user explicitly states their gender."
    )
    urgency: Literal["immediate", "high", "normal", "low"] = "normal"
    domain: Literal[
        "rent", "consumer", "wages", "legal_notice", "general", "uncertain"
    ] = Field(
        description=(
            "Use uncertain when the facts do not support a confident classification. "
            "Use general only for a known issue outside supported domains."
        )
    )
    domain_confidence: Literal["high", "low"]
    extraction_confidence: Literal["high", "low"]
    action_needed: Literal[
        "legal_notice", "complaint", "reply", "application", "advice"
    ]
    clarification_needed: bool
    clarification_question: str = Field(
        default="",
        description="One focused question when clarification is needed; otherwise empty."
    )
