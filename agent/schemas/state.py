"""Shared LangGraph state contract for the legal-assistance workflow."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

TurnType = Literal[
    "new_problem",
    "followup",
    "modify_draft",
    "new_problem_same_session",
]


class BaseState(TypedDict, total=False):
    """State fields shared across workflow nodes.

    This is a partial TypedDict because LangGraph nodes return state patches.
    Conversation messages are managed by the checkpointer and are intentionally
    not duplicated here.
    """

    # Raw input
    user_message: str
    uploaded_pdf_text: str
    # LangGraph's reducer appends new turns; the checkpointer persists this
    # channel, so no separate hand-maintained conversation_history is needed.
    messages: Annotated[list[AnyMessage], add_messages]

    # Conversation/checkpoint identity
    thread_id: str
    turn_type: TurnType
    safety_route: Literal["self_harm", "imminent_threat", ""]

    # Supervisor extraction and routing
    issue_title: str
    problem_statement: str
    timeline_raw: str
    timeline_parsed: str
    proofs_stated: list[str]
    proofs_from_pdf: list[str]
    gender: Literal["male", "female", "unspecified"]
    urgency: Literal["immediate", "high", "normal", "low"]
    domain: Literal[
        "rent", "consumer", "wages", "legal_notice", "general", "uncertain"
    ]
    domain_confidence: Literal["high", "low"]
    extraction_confidence: Literal["high", "low"]
    action_needed: Literal[
        "legal_notice", "complaint", "reply", "application", "advice"
    ]
    clarification_needed: bool
    clarification_question: str

    # Domain analysis and retrieval provenance
    subagent_output: dict[str, object]
    sources: list[dict[str, object]]
    citation_audit_errors: list[str]
    fallback_used: bool
    limitation_status: Literal[
        "likely_within", "likely_expired", "unclear_verify_with_lawyer"
    ]
    limitation_detail: str

    # Downstream workflow results
    drafted_document: str
    send_to: str
    send_via: str
    compliance_flag: bool
    compliance_notes: str
    final_response: str
