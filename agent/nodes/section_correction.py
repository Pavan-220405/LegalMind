"""Correct unsupported statutory section numbers using retrieved metadata."""

import re

from agent.schemas.state import BaseState

_SECTION_REFERENCE = re.compile(
    r"\b(section|sec\.?|rule|regulation|article)\s+"
    r"(\d+[A-Za-z]?(?:[-/](?:\d+[A-Za-z]?))?(?:\(\d+[A-Za-z]?\))?)",
    re.IGNORECASE,
)


def _rewrite_references(
    value: str, supported: set[str], replacement: str | None
) -> tuple[str, bool]:
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        number = match.group(2)
        if number in supported:
            return match.group(0)
        changed = True
        if replacement is None:
            # Keep the legal point while removing a section number that the
            # retrieved material cannot verify.
            return f"{match.group(1)} cited in the retrieved material"
        return f"{match.group(1)} {replacement}"

    return _SECTION_REFERENCE.sub(replace, value), changed


def section_correction_node(state: BaseState) -> dict[str, object]:
    """Align section references in the analysis and draft to retrieved sources.

    When retrieval gives one distinct section, unsupported section numbers are
    corrected to it. If the evidence contains multiple or no sections, the
    unsupported number is removed instead of guessing.
    """

    supported = {
        str(source.get("section_number", "")).strip()
        for source in state.get("sources", [])
        if str(source.get("section_number", "")).strip()
    }
    replacement = next(iter(supported)) if len(supported) == 1 else None
    analysis = state.get("subagent_output", {})
    corrected_analysis = dict(analysis) if isinstance(analysis, dict) else {}
    corrections: list[str] = []

    for field in ("summary", "legal_information", "next_steps"):
        value = corrected_analysis.get(field)
        if isinstance(value, str):
            updated, changed = _rewrite_references(value, supported, replacement)
            corrected_analysis[field] = updated
            if changed:
                corrections.append(field)
        elif isinstance(value, list):
            updated_items = []
            changed_any = False
            for item in value:
                if isinstance(item, str):
                    updated, changed = _rewrite_references(
                        item, supported, replacement
                    )
                    updated_items.append(updated)
                    changed_any = changed_any or changed
                else:
                    updated_items.append(item)
            corrected_analysis[field] = updated_items
            if changed_any:
                corrections.append(field)

    citations = corrected_analysis.get("citations", [])
    if isinstance(citations, list):
        updated_citations = []
        citation_changed = False
        for citation in citations:
            if isinstance(citation, dict) and isinstance(citation.get("proposition"), str):
                updated = dict(citation)
                updated["proposition"], changed = _rewrite_references(
                    updated["proposition"], supported, replacement
                )
                citation_changed = citation_changed or changed
                updated_citations.append(updated)
            else:
                updated_citations.append(citation)
        corrected_analysis["citations"] = updated_citations
        if citation_changed:
            corrections.append("citations")

    drafted_document, draft_changed = _rewrite_references(
        state.get("drafted_document", ""), supported, replacement
    )
    if draft_changed:
        corrections.append("drafted_document")

    if not corrections:
        return {"section_correction_notes": ""}

    mode = (
        f"replaced unsupported section references with section {replacement}"
        if replacement is not None
        else "removed unsupported section numbers because retrieved sources did not identify one unambiguous section"
    )
    note = "Section correction: " + mode + "."
    return {
        "subagent_output": corrected_analysis,
        "drafted_document": drafted_document,
        "section_correction_notes": note,
        "compliance_notes": "\n".join(
            part for part in (state.get("compliance_notes", ""), note) if part
        ),
    }
