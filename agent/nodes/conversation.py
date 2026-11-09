"""Final response node using checkpointed LangGraph message history."""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.schemas.state import BaseState

CONVERSATION_SYSTEM_PROMPT = """
You are the conversation assistant for an Indian legal information workflow.
Use the current workflow findings and the checkpointed messages to answer the
latest user turn. Be clear about uncertainty. Do not claim to be a lawyer or
that a draft was sent or filed. Preserve the distinction between user-stated
evidence and PDF-extracted evidence. Mention urgent next steps and limitation
warnings when supplied. If the workflow asks a clarification question, ask it
plainly before giving routed advice. Keep the answer practical and concise.
""".strip()


def _current_user_turn_already_recorded(state: BaseState) -> bool:
    messages = state.get("messages", [])
    current_text = state.get("user_message", "")
    return bool(
        messages
        and isinstance(messages[-1], HumanMessage)
        and messages[-1].content == current_text
    )


def record_user_turn(state: BaseState) -> dict[str, object]:
    message = state.get("user_message", "")
    patch: dict[str, object] = {
        # Checkpointed messages are intentionally preserved by their reducer;
        # all other values below belong only to the current turn.
        "issue_title": "",
        "problem_statement": "",
        "timeline_raw": "",
        "timeline_parsed": "",
        "proofs_stated": [],
        "proofs_from_pdf": [],
        "gender": "unspecified",
        "urgency": "normal",
        "domain": "uncertain",
        "domain_confidence": "low",
        "extraction_confidence": "low",
        "action_needed": "advice",
        "clarification_needed": False,
        "clarification_question": "",
        "subagent_output": {},
        "sources": [],
        "citation_audit_errors": [],
        "fallback_used": False,
        "limitation_status": "unclear_verify_with_lawyer",
        "limitation_detail": "",
        "drafted_document": "",
        "send_to": "",
        "send_via": "",
        "compliance_flag": False,
        "compliance_notes": "",
        "final_response": "",
        "safety_route": "",
    }
    if message and not _current_user_turn_already_recorded(state):
        patch["messages"] = [HumanMessage(content=message)]
    return patch


def build_conversation_node(chat_model: Any) -> Any:
    def conversation_node(state: BaseState) -> dict[str, object]:
        if state.get("clarification_needed"):
            response = state.get("clarification_question", "") or (
                "Could you share a little more detail so I can identify the right legal area?"
            )
        elif state.get("fallback_used"):
            output = state.get("subagent_output", {})
            response = str(output.get("summary", "")).strip()
            response += (
                "\n\nThis is general information, not legal advice. For advice on "
                "your location and circumstances, consult a qualified local lawyer."
            )
            if state.get("urgency") in {"immediate", "high"}:
                response += (
                    "\n\nBecause this may be urgent, contact the appropriate local "
                    "authority or a lawyer promptly."
                )
        else:
            output = state.get("subagent_output", {})
            # Keep every message in the checkpoint, but bound prompt size on
            # long-running threads. Sixteen messages retain the latest eight turns.
            recent_messages = state.get("messages", [])[-16:]
            context = {
                "analysis": output,
                "drafted_document": state.get("drafted_document", ""),
                "send_to": state.get("send_to", ""),
                "send_via": state.get("send_via", ""),
                "compliance_flag": state.get("compliance_flag", False),
                "compliance_notes": state.get("compliance_notes", ""),
                "limitation_status": state.get("limitation_status", ""),
                "limitation_detail": state.get("limitation_detail", ""),
                "sources": state.get("sources", []),
            }
            response_message = chat_model.invoke(
                [
                    SystemMessage(content=CONVERSATION_SYSTEM_PROMPT),
                    *recent_messages,
                    SystemMessage(content=f"Workflow findings for this turn:\n{context}"),
                ]
            )
            response = str(getattr(response_message, "content", response_message))

        return {"final_response": response, "messages": [AIMessage(content=response)]}

    return conversation_node
