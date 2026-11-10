"""Structured result returned by the document-drafting node."""

from pydantic import BaseModel, Field


class DraftOutput(BaseModel):
    drafted_document: str = Field(
        default="",
        description="The requested draft, with clear placeholders for unknown facts."
    )
    send_to: str = Field(
        default="[Recipient]",
        description="Intended recipient, or a placeholder if unknown.",
    )
    send_via: str = Field(
        default="Confirm an appropriate written channel",
        description="Suggested channel, or state that the channel must be confirmed."
    )
