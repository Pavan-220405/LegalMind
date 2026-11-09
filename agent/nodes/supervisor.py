"""Supervisor node that extracts intake facts using structured output."""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT
from agent.nodes.structured import bind_structured_output
from agent.nodes.safety import is_crisis_support_response
from agent.schemas.state import BaseState
from agent.schemas.supervisor import SupervisorOutput


def build_supervisor_node(chat_model: Any) -> Callable[[BaseState], dict[str, object]]:
    """Bind a chat model to the supervisor schema and return a graph node.

    The model is injected by the application, which keeps provider credentials
    and model selection outside the workflow node.
    """

    structured_model = bind_structured_output(chat_model, SupervisorOutput)

    def supervisor_node(state: BaseState) -> dict[str, object]:
        user_message = state.get("user_message", "")
        pdf_text = state.get("uploaded_pdf_text", "")
        history = state.get("messages", [])
        if (
            history
            and getattr(history[-1], "type", "") == "human"
            and getattr(history[-1], "content", None) == user_message
        ):
            history = history[:-1]
        recent_context = "\n".join(
            f"{getattr(message, 'type', 'message')}: "
            f"{str(getattr(message, 'content', message))[:1500]}"
            for message in history[-8:]
        ) or "[no earlier turns]"
        try:
            response = structured_model.invoke(
                [
                    SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"Turn type: {state.get('turn_type', 'new_problem')}\n"
                            f"Recent checkpointed conversation (prior turns only):\n"
                            f"{recent_context}\n\n"
                            f"User message:\n{user_message or '[none provided]'}\n\n"
                            f"Uploaded PDF text:\n{pdf_text or '[none provided]'}"
                        )
                    ),
                ]
            )
        except Exception as exc:
            # Groq may emit a plain-text safety response instead of JSON for a
            # self-harm disclosure. Recover only that recognizable case; let
            # unrelated provider/schema errors surface normally.
            body = getattr(exc, "body", None)
            error = body.get("error", {}) if isinstance(body, dict) else {}
            failed_generation = str(error.get("failed_generation", ""))
            if (
                error.get("code") == "json_validate_failed"
                and is_crisis_support_response(failed_generation)
            ):
                return {"safety_route": "self_harm"}
            raise

        if isinstance(response, SupervisorOutput):
            output = response
        elif isinstance(response, dict):
            output = SupervisorOutput.model_validate(response)
        else:
            raise TypeError(
                "Structured supervisor model must return SupervisorOutput or a mapping; "
                f"received {type(response).__name__}."
            )

        fields = output.model_dump()
        if (
            output.domain == "uncertain"
            or output.domain_confidence == "low"
            or (
                output.extraction_confidence == "low"
                and output.domain_confidence == "low"
            )
        ):
            fields["clarification_needed"] = True
            if not fields["clarification_question"].strip():
                fields["clarification_question"] = (
                    "Could you clarify what kind of legal issue this is and what outcome you need?"
                )
        return fields

    return supervisor_node
