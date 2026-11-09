"""Conservative fallback node for known unsupported domains."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts.domain import GENERAL_FALLBACK_PROMPT
from agent.schemas.domain import SubAgentOutput
from agent.schemas.state import BaseState


def build_general_fallback_agent(chat_model: Any) -> Any:
    def general_fallback_agent(state: BaseState) -> dict[str, object]:
        response = chat_model.invoke(
            [
                SystemMessage(content=GENERAL_FALLBACK_PROMPT),
                HumanMessage(
                    content=(
                        f"Issue: {state.get('problem_statement', '')}\n"
                        f"User message: {state.get('user_message', '')}\n"
                        f"Jurisdiction stated: not provided to this node."
                    )
                ),
            ]
        )
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        output = SubAgentOutput(
            summary=str(content).strip(),
            legal_information=[],
            next_steps=[],
            citations=[],
        )
        return {
            "subagent_output": output.model_dump(),
            "sources": [],
            "fallback_used": True,
        }

    return general_fallback_agent
