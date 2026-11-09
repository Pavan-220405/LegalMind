"""Structured analysis and citation data returned by domain agents."""

from pydantic import BaseModel, Field


class LegalCitation(BaseModel):
    source_id: str = Field(description="An ID from the supplied source list, such as S1.")
    proposition: str = Field(description="The legal point supported by this source.")


class SubAgentOutput(BaseModel):
    summary: str
    legal_information: list[str] = Field(
        default_factory=list,
        description="Relevant law supported by supplied sources; empty when none applies.",
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="Practical next steps; empty when none can be supported.",
    )
    citations: list[LegalCitation] = Field(
        default_factory=list,
        description="Retrieved source references; empty when no source supports the answer.",
    )
