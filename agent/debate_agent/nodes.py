"""Message-history supervisor and three retrieval-grounded vertical nodes."""

from collections.abc import Callable
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from agent.nodes.structured import bind_structured_output
from agent.retriever.config import RERANK_TOP_K

from .prompts import COACH_PROMPT, DEBATER_PROMPT, SUPERVISOR_PROMPT
from .schemas import DebateSupervisorOutput
from .state import DebateState

SUPPORTED_DOMAINS = {"rent", "consumer", "wages"}
MAX_CONTEXT_CHARS = 16_000
MAX_RETRIEVAL_QUERY_CHARS = 12_000


def _bounded(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    marker = f"\n[{label} truncated]"
    return text[: max(0, limit - len(marker))] + marker[:limit]


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _content(message: Any) -> str:
    return str(getattr(message, "content", message))


def _affirmative(text: str) -> bool:
    return bool(
        re.match(
            r"^(yes|yeah|yep|correct|that's right|that’s right|right|confirmed|"
            r"let's start|let’s start|start|go ahead)(?:\b|[,!.])",
            text.casefold().strip(),
        )
    )


def _correction(text: str) -> bool:
    return bool(re.search(r"\b(no|not right|instead|actually|change that)\b", text.casefold()))


def _clarification(
    *, topic: str, domain: str, position: str, jurisdiction: str,
    model_question: str, confirmation_pending: bool,
) -> str:
    if domain not in SUPPORTED_DOMAINS:
        return model_question or (
            "I can currently practise rent and tenancy, consumer protection, or "
            "wages issues. Which area best fits your situation?"
        )
    missing = []
    if not topic:
        missing.append("the specific issue you want to practise")
    if not position:
        missing.append("the side or position you want to argue")
    if missing:
        return "Before we start, could you clarify " + " and ".join(missing) + "?"
    if confirmation_pending:
        lead = "Please confirm the updated setup"
    else:
        if model_question:
            return model_question
        lead = "Before we start, please confirm"
    return (
        f"{lead}: you want to practise “{topic}” in {jurisdiction or 'India'}, "
        f"you’ll argue “{position},” and I’ll take the opposing side. Is that right?"
    )


def _invoke_supervisor(model: Any, state: DebateState) -> DebateSupervisorOutput:
    messages = state.get("messages", [])
    recent = "\n".join(
        f"{getattr(message, 'type', 'message')}: {_content(message)[:1600]}"
        for message in messages[-12:]
    )
    try:
        result = model.invoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=f"Checkpointed conversation history:\n{recent or '[none]'}"),
            ]
        )
    except (OutputParserException, ValidationError):
        return DebateSupervisorOutput()
    if isinstance(result, DebateSupervisorOutput):
        return result
    try:
        if isinstance(result, dict):
            return DebateSupervisorOutput.model_validate(result)
        if hasattr(result, "model_dump"):
            return DebateSupervisorOutput.model_validate(result.model_dump())
    except (ValidationError, TypeError):
        pass
    return DebateSupervisorOutput()


def build_debate_supervisor_node(model: Any) -> Callable[[DebateState], dict[str, object]]:
    structured_model = bind_structured_output(model, DebateSupervisorOutput)

    def supervisor_node(state: DebateState) -> dict[str, object]:
        output = _invoke_supervisor(structured_model, state)
        messages = state.get("messages", [])
        prior = messages[:-1] if messages and getattr(messages[-1], "type", "") == "human" else messages
        prior_assistant_text = next(
            (_content(message) for message in reversed(prior) if getattr(message, "type", "") == "ai"),
            "",
        )
        first_turn = not any(getattr(message, "type", "") == "ai" for message in prior)
        confirmation_pending = bool(
            re.search(r"please confirm|is that right", prior_assistant_text.casefold())
        )
        latest = _text(state.get("user_message"))

        topic = _text(output.topic) or _text(state.get("practice_topic"))
        raw_domain = _text(output.domain)
        if raw_domain in SUPPORTED_DOMAINS:
            domain = raw_domain
        elif raw_domain in {"general", "uncertain"}:
            domain = ""
        else:
            domain = _text(state.get("practice_domain"))
        position = _text(output.user_position) or _text(state.get("practice_user_position"))
        jurisdiction = _text(output.jurisdiction) or _text(state.get("practice_jurisdiction"))
        complete = bool(
            domain in SUPPORTED_DOMAINS
            and topic
            and position
        )

        if first_turn:
            ready = False
        elif confirmation_pending:
            ready = complete and not _correction(latest) and (
                output.ready is True or _affirmative(latest)
            )
        else:
            ready = complete and output.ready is not False and not _text(
                output.clarification_question
            )

        question = ""
        if not ready:
            question = _clarification(
                topic=topic,
                domain=domain,
                position=position,
                jurisdiction=jurisdiction,
                model_question=_text(output.clarification_question),
                confirmation_pending=confirmation_pending,
            )

        latest_lower = latest.casefold()
        declines_coaching = bool(
            re.search(r"\b(don't|do not|dont)\s+coach\b", latest_lower)
        )
        coaching_requested = not declines_coaching and (
            output.request_coaching is True
            or bool(re.search(r"\b(coach|coaching|feedback|review my argument)\b", latest_lower))
        )

        patch: dict[str, object] = {
            "practice_topic": topic,
            "practice_domain": domain,
            "practice_user_position": position,
            "practice_jurisdiction": jurisdiction,
            "practice_ready": ready,
            "practice_request_coaching": coaching_requested,
            "domain": domain if domain in SUPPORTED_DOMAINS else "uncertain",
            "problem_statement": topic,
            "clarification_needed": not ready,
            "clarification_question": question,
        }
        if not ready:
            patch["final_response"] = question
            patch["messages"] = [AIMessage(content=question)]
        return patch

    return supervisor_node


def _format_sources(documents: list[Document]) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    sources: list[dict[str, object]] = []
    contexts: list[dict[str, object]] = []
    excerpts: list[str] = []
    total = 0
    for index, document in enumerate(documents[: min(6, RERANK_TOP_K)], start=1):
        remaining = MAX_CONTEXT_CHARS - total
        if remaining <= 0:
            break
        content = document.page_content[: min(6_000, remaining)]
        metadata = document.metadata or {}
        source = {
            "source_id": f"S{index}",
            "title": str(metadata.get("document_name") or metadata.get("source_path") or "Legal source"),
            "section_number": str(metadata.get("section_number") or ""),
            "section_title": str(metadata.get("section_title") or ""),
            "page_start": metadata.get("page_start"),
            "page_end": metadata.get("page_end"),
            "source_path": str(metadata.get("source_path") or ""),
        }
        sources.append(source)
        contexts.append({**source, "content": content})
        excerpts.append(
            f"[{source['source_id']}] {source['title']}; section {source['section_number']}; "
            f"pages {source['page_start']}-{source['page_end']}\n{content}"
        )
        total += len(content)
    return sources, contexts, "\n\n".join(excerpts)


def build_vertical_node(
    collection: str,
    display_name: str,
    model: Any,
    retrieve_fn: Callable[..., list[Document]],
) -> Callable[[DebateState], dict[str, object]]:
    def vertical_node(state: DebateState) -> dict[str, object]:
        if state.get("practice_request_coaching") and state.get("practice_started"):
            arguments = [
                _content(message)
                for message in state.get("messages", [])
                if getattr(message, "type", "") == "human"
            ][-5:]
            excerpts = "\n\n".join(
                f"[{item.get('source_id', 'S?')}] {item.get('title', 'Legal source')}\n"
                f"{item.get('content', '')}"
                for item in state.get("practice_retrieved_contexts", [])
                if isinstance(item, dict)
            )
            answer = model.invoke(
                [
                    SystemMessage(content=COACH_PROMPT),
                    HumanMessage(
                        content=(
                            f"Issue: {_text(state.get('practice_topic'))}\n"
                            f"User's position: {_text(state.get('practice_user_position'))}\n"
                            f"Recent user turns:\n{chr(10).join(arguments)}\n\n"
                            f"Retrieved excerpts:\n{excerpts or '[no excerpts available]'}"
                        )
                    ),
                ]
            )
            response = str(getattr(answer, "content", answer)).strip() or (
                "I couldn’t prepare feedback this turn. Please restate the argument you’d like me to review."
            )
            return {
                "practice_last_argument": response,
                "final_response": response,
                "messages": [AIMessage(content=response)],
            }

        topic = _text(state.get("practice_topic"))
        query = _bounded(
            "\n".join(
                part
                for part in (
                    topic,
                    "User's position: " + _text(state.get("practice_user_position")),
                    "Latest point: " + _text(state.get("user_message")),
                )
                if part.strip()
            ),
            MAX_RETRIEVAL_QUERY_CHARS,
            label="Debate retrieval query",
        )
        documents = (retrieve_fn(query, collection) if query else []) or []
        sources, contexts, excerpts = _format_sources(documents)
        if not excerpts:
            response = (
                f"I couldn’t retrieve relevant legal sources in the {display_name} collection, "
                "so I can’t make a law-based opposing argument yet. You can narrow the issue "
                "or share the legal point you want to practise."
            )
        else:
            history = "\n".join(
                f"{getattr(message, 'type', 'message')}: {_content(message)[:1400]}"
                for message in state.get("messages", [])[-8:]
            )
            history = _bounded(history, 8_000, label="Debate history")
            answer = model.invoke(
                [
                    SystemMessage(content=DEBATER_PROMPT),
                    HumanMessage(
                        content=(
                            f"Practice issue: {topic}\n"
                            f"Jurisdiction: {_text(state.get('practice_jurisdiction')) or 'India'}\n"
                            f"Your role: opposing side to {_text(state.get('practice_user_position'))}\n"
                            f"Recent exchange:\n{history or '[none]'}\n"
                            f"Retrieved excerpts:\n{excerpts}\n\n"
                            "Make one concise opposing argument, cite supporting excerpts, and invite the user to respond."
                        )
                    ),
                ]
            )
            response = str(getattr(answer, "content", answer)).strip() or (
                "I’m ready to continue the practice, but couldn’t produce an argument this turn."
            )
        return {
            "practice_domain": collection,
            "practice_started": True,
            "practice_request_coaching": False,
            "practice_retrieved_contexts": contexts,
            "practice_last_argument": response,
            "sources": sources,
            "clarification_needed": False,
            "clarification_question": "",
            "final_response": response,
            "messages": [AIMessage(content=response)],
        }

    return vertical_node
