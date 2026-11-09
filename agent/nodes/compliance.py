"""Phase-one checks for citations, limitation uncertainty, and urgency."""

from datetime import date, datetime, timezone
from typing import Any

from agent.schemas.state import BaseState

_EXPLICIT_DATE_FORMATS = (
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
)


def _explicit_date(value: str) -> date | None:
    value = value.strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        pass
    for fmt in _EXPLICIT_DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _limitation_assessment(state: BaseState) -> tuple[str, str]:
    domain = state.get("domain", "general")
    event_date = _explicit_date(state.get("timeline_parsed", ""))
    today = datetime.now(timezone.utc).date()

    if domain == "consumer":
        detail = (
            "Consumer Protection Act, 2019, section 69 generally sets a two-year "
            "period from the cause of action for a consumer complaint. The law "
            "allows delay to be condoned for sufficient cause, and the relevant "
            "cause-of-action date and forum must be checked by a lawyer. This is "
            "a heuristic, not a limitation conclusion."
        )
        if event_date is None:
            return "unclear_verify_with_lawyer", detail
        elapsed_days = (today - event_date).days
        if elapsed_days < 0:
            return "unclear_verify_with_lawyer", detail
        if elapsed_days <= 2 * 366:
            return "likely_within", detail
        return "likely_expired", detail

    if domain == "wages":
        return (
            "unclear_verify_with_lawyer",
            "Wage-claim deadlines depend on the applicable law, commencement, "
            "claim type, and facts. No fixed period is inferred here; verify "
            "promptly with a lawyer or the appropriate labour authority.",
        )
    if domain == "rent":
        return (
            "unclear_verify_with_lawyer",
            "Rent and tenancy deadlines can depend on the state, agreement, "
            "claim type, and cause-of-action date. No fixed period is inferred "
            "here; verify promptly with a local lawyer.",
        )
    return (
        "unclear_verify_with_lawyer",
        "The applicable limitation period cannot be assessed for this issue "
        "without confirming the claim type, jurisdiction, and cause-of-action date.",
    )


def compliance_node(state: BaseState) -> dict[str, object]:
    notes: list[str] = []
    flagged = False
    sources = state.get("sources", [])
    valid_ids = {str(source.get("source_id", "")) for source in sources}
    analysis = state.get("subagent_output", {})
    citations = analysis.get("citations", []) if isinstance(analysis, dict) else []
    invalid = [
        citation.get("source_id", "")
        for citation in citations
        if str(citation.get("source_id", "")) not in valid_ids
    ]
    recorded_errors = state.get("citation_audit_errors", [])
    if invalid or recorded_errors:
        flagged = True
        notes.append(
            "Citation audit: one or more citations do not match retrieved sources."
        )
    elif citations:
        notes.append("Citation audit: cited source IDs match retrieved legal excerpts.")
    elif analysis.get("legal_information"):
        flagged = True
        notes.append("Citation audit: legal information has no retrieved source citations.")

    limitation_status, limitation_detail = _limitation_assessment(state)
    notes.append(f"Limitation warning ({limitation_status}): {limitation_detail}")
    if limitation_status == "unclear_verify_with_lawyer":
        flagged = True

    urgency = state.get("urgency", "normal")
    if urgency in {"immediate", "high"}:
        flagged = True
        notes.append(
            "Urgency flag: the user described a time-sensitive or potentially urgent "
            "situation; advise prompt contact with the appropriate local authority or lawyer."
        )
    else:
        notes.append(f"Urgency flag: {urgency}; no immediate escalation signal was classified.")

    if state.get("domain") == "general" or state.get("fallback_used"):
        notes.append(
            "General fallback disclaimer: this system has no verified domain-specific "
            "legal source for this issue; the response is general information, not legal advice."
        )

    return {
        "compliance_flag": flagged,
        "compliance_notes": "\n".join(notes),
        "limitation_status": limitation_status,
        "limitation_detail": limitation_detail,
    }
