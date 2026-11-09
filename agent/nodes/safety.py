"""Deterministic safety routing for explicit imminent harm disclosures."""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import AIMessage

from agent.schemas.state import BaseState

SafetyRoute = Literal["self_harm", "imminent_threat", ""]

_SELF_HARM_PATTERNS = (
    r"\b(?:i|we)\s+(?:want|plan|intend|am going)\s+to\s+(?:kill|harm|hurt)\s+myself\b",
    r"\b(?:i|we)\s+(?:want|plan|intend|am going)\s+to\s+(?:commit\s+)?suicide\b",
    r"\b(?:i|we)\s+(?:might|may|will|could)\s+(?:kill|hurt|harm)\s+myself\b",
    r"\b(?:i|we)\s+(?:(?:have\s+)?(?:tried|attempted)|(?:am|are|'m)\s+(?:trying|attempting))\s+suicide\b",
    r"\b(?:i|we)\s+(?:feel|am|'m)\s+suicidal\b",
    r"\b(?:i|we)\s+(?:want|wanna)\s+to\s+die\b",
    r"\b(?:i|we)\s+(?:do\s+not|don't|dont)\s+want\s+to\s+live\b",
    r"\b(?:my\s+(?:friend|brother|sister|child|parent|partner)|he|she|they)\s+(?:is|are|'s)\s+(?:suicidal|thinking\s+about\s+(?:suicide|ending\s+(?:their|his|her)\s+life))\b",
    r"\b(?:my\s+(?:friend|brother|sister|child|parent|partner)|he|she|they)\s+(?:wants?|plans?|is\s+trying|has\s+tried|attempted)\s+to\s+(?:commit\s+)?suicide\b",
    r"\b(?:suicidal|suicide attempt|attempted suicide|self[- ]harm|self[- ]harming)\b",
    r"\b(?:i|we)\s+(?:can't|cannot|can not)\s+go\s+on\b",
    r"\b(?:there is|there's)\s+no\s+(?:point|reason)\s+(?:in\s+)?living\b",
    r"\b(?:i|we)\s+(?:just\s+)?(?:overdosed|cut myself|hanged myself)\b",
    r"\b(?:suicidal thoughts|thinking about suicide|end my life|take my own life)\b",
)

_IMMINENT_THREAT_PATTERNS = (
    r"\b(?:i|we)\s+(?:am|are|'m)\s+(?:going\s+to|about\s+to)\s+(?:kill|shoot|stab|attack)\s+(?:him|her|them|someone|my|the)\b",
    r"\b(?:i|we)\s+(?:will|plan to|intend to)\s+(?:kill|shoot|stab|attack)\s+(?:him|her|them|someone|my|the)\b",
)


def detect_safety_route(text: str) -> SafetyRoute:
    """Recognize direct self-harm disclosures or immediate threats of violence."""

    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    if any(re.search(pattern, normalized) for pattern in _SELF_HARM_PATTERNS):
        return "self_harm"
    if any(re.search(pattern, normalized) for pattern in _IMMINENT_THREAT_PATTERNS):
        return "imminent_threat"
    return ""


def is_crisis_support_response(text: str) -> bool:
    """Identify provider safety prose returned instead of requested JSON."""

    normalized = text.casefold()
    crisis_terms = ("suicid", "self-harm", "988", "crisis helpline")
    support_terms = (
        "immediate danger",
        "emergency services",
        "trusted friend",
        "you deserve help",
        "you are not alone",
        "you don't have to face",
    )
    return any(term in normalized for term in crisis_terms) and any(
        term in normalized for term in support_terms
    )


def safety_response_node(state: BaseState) -> dict[str, object]:
    """Return concise crisis resources without sending sensitive text to an LLM."""

    route = state.get("safety_route", "")
    if route == "self_harm":
        response = (
            "I’m sorry you’re facing this. If you may hurt yourself now, or have "
            "already taken steps to do so, move away from anything you could use "
            "to harm yourself and contact someone you trust who can stay with you. "
            "In India, call 112 for immediate emergency help. You can also call "
            "Tele-MANAS at 14416 (or 1-800-891-4416) for mental-health support. "
            "If you are elsewhere, contact your local emergency number or crisis "
            "service. Are you in immediate danger right now?"
        )
    else:
        response = (
            "If someone may be harmed now, create distance from that person and "
            "from any weapons, and contact emergency services immediately. In "
            "India, call 112; elsewhere, call your local emergency number. If you "
            "can, tell a trusted person nearby and ask them to stay with you while "
            "you get help. Is anyone in immediate danger right now?"
        )
    return {"final_response": response, "messages": [AIMessage(content=response)]}
