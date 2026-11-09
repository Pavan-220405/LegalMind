"""Node for preparing reviewable legal document drafts."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.nodes.structured import bind_structured_output
from agent.prompts.drafting import DRAFTING_PROMPT
from agent.schemas.drafting import DraftOutput
from agent.schemas.state import BaseState


def build_drafting_node(chat_model: Any) -> Any:
    structured_model = bind_structured_output(chat_model, DraftOutput)

    def drafting_node(state: BaseState) -> dict[str, object]:
        action = state.get("action_needed", "advice")
        if action == "advice":
            return {"drafted_document": "", "send_to": "", "send_via": ""}

        facts = {
            "issue_title": state.get("issue_title", ""),
            "problem_statement": state.get("problem_statement", ""),
            "user_message": state.get("user_message", ""),
            "timeline_raw": state.get("timeline_raw", ""),
            "proofs_stated": state.get("proofs_stated", []),
            "proofs_from_pdf": state.get("proofs_from_pdf", []),
            "action_needed": action,
            "legal_analysis": state.get("subagent_output", {}),
        }
        response = structured_model.invoke(
            [
                SystemMessage(content=DRAFTING_PROMPT),
                HumanMessage(content=f"Prepare this draft from the following data:\n{facts}"),
            ]
        )
        output = (
            response
            if isinstance(response, DraftOutput)
            else DraftOutput.model_validate(response)
        )
        return output.model_dump()

    return drafting_node
