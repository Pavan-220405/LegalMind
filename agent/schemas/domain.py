"""Structured analysis and citation data returned by domain agents."""

from pydantic import BaseModel, ConfigDict, Field


class LegalCitation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_id: str | None = Field(
        default=None,
        description="An ID from the supplied source list, such as S1.",
    )
    proposition: str | None = Field(
        default=None,
        description="The legal point supported by this source.",
    )


class SubAgentOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str | None = ""
    legal_information: list[str] | None = Field(
        default_factory=list,
        description="Relevant law supported by supplied sources; empty when none applies.",
    )
    next_steps: list[str] | None = Field(
        default_factory=list,
        description="Practical next steps; empty when none can be supported.",
    )
    citations: list[LegalCitation] | None = Field(
        default_factory=list,
        description="Retrieved source references; empty when no source supports the answer.",
    )
