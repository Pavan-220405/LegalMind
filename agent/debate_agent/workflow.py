"""Minimal debate graph: safety router, message-history supervisor, 3 verticals."""

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes.conversation import record_user_turn
from agent.nodes.safety import detect_safety_route, safety_response_node
from agent.retriever.retrieval import retrieve_documents

from .nodes import build_debate_supervisor_node, build_vertical_node
from .state import DebateState

_VERTICALS = {
    "rent": ("rent_debate", "rent and tenancy"),
    "consumer": ("consumer_debate", "consumer protection"),
    "wages": ("wages_debate", "wages"),
}


def _record_turn(state: DebateState) -> dict[str, object]:
    patch = record_user_turn(state)
    # Generic turn reset clears legal-analysis sources; debate coaching needs
    # the excerpts from the previous debate turn, so retain the debate copy.
    patch["practice_retrieved_contexts"] = state.get("practice_retrieved_contexts", [])
    patch["sources"] = state.get("sources", [])
    return patch


def _route_after_record(state: DebateState) -> str:
    """Check safety in the router before the supervisor makes any model call."""
    user_message = state.get("user_message", "")
    if detect_safety_route(user_message if isinstance(user_message, str) else ""):
        return "safety_response"
    return "supervisor"


def _safety_response(state: DebateState) -> dict[str, object]:
    user_message = state.get("user_message", "")
    route = detect_safety_route(user_message if isinstance(user_message, str) else "")
    patch = safety_response_node({**state, "safety_route": route})
    patch["safety_route"] = route
    return patch


def _route_after_supervisor(state: DebateState) -> str:
    if state.get("safety_route"):
        return "safety_response"
    if not state.get("practice_ready"):
        # Supervisor has already added its clarification to message history.
        return END
    return _VERTICALS.get(state.get("practice_domain", ""), (END, ""))[0]


def build_debate_workflow(
    *,
    supervisor_model: Any,
    debate_model: Any | None = None,
    retrieve_fn: Any = retrieve_documents,
    checkpointer: Any | None = None,
) -> Any:
    """Build Debate Practice with conditional routing to three vertical nodes."""

    debate_model = debate_model or supervisor_model
    graph = StateGraph(DebateState)
    graph.add_node("record_user_turn", _record_turn)
    graph.add_node("safety_response", _safety_response)
    graph.add_node("supervisor", build_debate_supervisor_node(supervisor_model))
    graph.add_node(
        "rent_debate",
        build_vertical_node("rent", "rent and tenancy", debate_model, retrieve_fn),
    )
    graph.add_node(
        "consumer_debate",
        build_vertical_node("consumer", "consumer protection", debate_model, retrieve_fn),
    )
    graph.add_node(
        "wages_debate",
        build_vertical_node("wages", "wages", debate_model, retrieve_fn),
    )

    graph.add_edge(START, "record_user_turn")
    graph.add_conditional_edges(
        "record_user_turn",
        _route_after_record,
        {"safety_response": "safety_response", "supervisor": "supervisor"},
    )
    graph.add_edge("safety_response", END)
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "safety_response": "safety_response",
            "rent_debate": "rent_debate",
            "consumer_debate": "consumer_debate",
            "wages_debate": "wages_debate",
            END: END,
        },
    )
    for node_name, _display in _VERTICALS.values():
        graph.add_edge(node_name, END)
    return graph.compile(checkpointer=MemorySaver() if checkpointer is None else checkpointer)


def invoke_debate_turn(
    graph: Any,
    *,
    thread_id: str,
    user_message: str,
    turn_type: str = "new_problem",
) -> dict[str, Any]:
    if not thread_id.strip():
        raise ValueError("thread_id must be non-empty for checkpointed debate sessions")
    return graph.invoke(
        {"thread_id": thread_id, "user_message": user_message, "turn_type": turn_type},
        config={"configurable": {"thread_id": thread_id}},
    )


def stream_debate_turn(
    graph: Any,
    *,
    thread_id: str,
    user_message: str,
    turn_type: str = "new_problem",
) -> Any:
    if not thread_id.strip():
        raise ValueError("thread_id must be non-empty for checkpointed debate sessions")
    return graph.stream(
        {"thread_id": thread_id, "user_message": user_message, "turn_type": turn_type},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="debug",
    )
