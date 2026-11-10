"""LangGraph assembly, conditional routing, and PostgreSQL persistence."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langgraph.graph import END, START, StateGraph

from config import settings

from agent.nodes.compliance import compliance_node
from agent.nodes.consumer import build_consumer_agent
from agent.nodes.conversation import build_conversation_node, record_user_turn
from agent.nodes.drafting import build_drafting_node
from agent.nodes.general import build_general_fallback_agent
from agent.nodes.rent import build_rent_agent
from agent.nodes.safety import detect_safety_route, safety_response_node
from agent.nodes.section_correction import section_correction_node
from agent.nodes.supervisor import build_supervisor_node
from agent.nodes.wages import build_wages_agent
from agent.retriever.retrieval import retrieve_documents
from agent.schemas.state import BaseState


def _route_after_supervisor(state: BaseState) -> str:
    if state.get("safety_route"):
        return "safety_response"
    if state.get("clarification_needed") or state.get("domain") == "uncertain":
        return "conversation"
    return {
        "rent": "rent",
        "consumer": "consumer",
        "wages": "wages",
        "general": "general",
        # A legal-notice request without a supported legal domain follows the
        # fallback path, which keeps the draft generic and caveated.
        "legal_notice": "general",
    }.get(state.get("domain", "uncertain"), "conversation")


def _classify_safety(state: BaseState) -> dict[str, object]:
    """Check the current user message locally before invoking any model."""

    return {"safety_route": detect_safety_route(state.get("user_message", ""))}


def _route_after_safety_check(state: BaseState) -> str:
    return "safety_response" if state.get("safety_route") else "supervisor"


def build_legal_workflow(
    *,
    supervisor_model: Any,
    domain_model: Any | None = None,
    drafting_model: Any | None = None,
    conversation_model: Any | None = None,
    retrieve_fn: Any = retrieve_documents,
    checkpointer: Any | None = None,
) -> Any:
    """Build and compile the full workflow with injectable integrations."""

    domain_model = domain_model or supervisor_model
    drafting_model = drafting_model or domain_model
    conversation_model = conversation_model or domain_model

    graph = StateGraph(BaseState)
    graph.add_node("record_user_turn", record_user_turn)
    graph.add_node("safety_check", _classify_safety)
    graph.add_node("safety_response", safety_response_node)
    graph.add_node("supervisor", build_supervisor_node(supervisor_model))
    graph.add_node("rent", build_rent_agent(domain_model, retrieve_fn))
    graph.add_node("consumer", build_consumer_agent(domain_model, retrieve_fn))
    graph.add_node("wages", build_wages_agent(domain_model, retrieve_fn))
    graph.add_node("general", build_general_fallback_agent(domain_model))
    graph.add_node("drafting", build_drafting_node(drafting_model))
    graph.add_node("compliance", compliance_node)
    graph.add_node("section_correction", section_correction_node)
    graph.add_node("conversation", build_conversation_node(conversation_model))

    graph.add_edge(START, "record_user_turn")
    graph.add_edge("record_user_turn", "safety_check")
    graph.add_conditional_edges(
        "safety_check",
        _route_after_safety_check,
        {"safety_response": "safety_response", "supervisor": "supervisor"},
    )
    graph.add_edge("safety_response", END)
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "rent": "rent",
            "consumer": "consumer",
            "wages": "wages",
            "general": "general",
            "safety_response": "safety_response",
            "conversation": "conversation",
        },
    )
    for domain_node in ("rent", "consumer", "wages", "general"):
        graph.add_edge(domain_node, "drafting")
    graph.add_edge("drafting", "compliance")
    graph.add_edge("compliance", "section_correction")
    graph.add_edge("section_correction", "conversation")
    graph.add_edge("conversation", END)
    return graph.compile(checkpointer=checkpointer)


def invoke_turn(
    graph: Any,
    *,
    thread_id: str,
    user_message: str,
    uploaded_pdf_text: str = "",
    turn_type: str = "new_problem",
) -> dict[str, Any]:
    """Invoke a turn using the thread ID stored by the checkpointer."""

    if not thread_id.strip():
        raise ValueError("thread_id must be non-empty for checkpointed conversations")
    return graph.invoke(
        {
            "thread_id": thread_id,
            "user_message": user_message,
            "uploaded_pdf_text": uploaded_pdf_text,
            "turn_type": turn_type,
        },
        config={"configurable": {"thread_id": thread_id}},
    )


def stream_turn(
    graph: Any,
    *,
    thread_id: str,
    user_message: str,
    uploaded_pdf_text: str = "",
    turn_type: str = "new_problem",
) -> Any:
    """Stream graph task lifecycle events for a visible pipeline trace."""

    if not thread_id.strip():
        raise ValueError("thread_id must be non-empty for checkpointed conversations")
    return graph.stream(
        {
            "thread_id": thread_id,
            "user_message": user_message,
            "uploaded_pdf_text": uploaded_pdf_text,
            "turn_type": turn_type,
        },
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="debug",
    )


def create_chat_model() -> Any:
    """Create the configured production chat model from environment settings."""

    provider = settings.legal_rag_llm_provider
    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise RuntimeError("Install langchain-groq to use Groq") from exc
        return ChatGroq(
            model=settings.legal_model,
            temperature=settings.legal_model_temperature,
            api_key=settings.groq_api_key,
        )
    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install langchain-google-genai to use Google") from exc
        return ChatGoogleGenerativeAI(
            model=settings.legal_model,
            temperature=settings.legal_model_temperature,
            google_api_key=settings.google_api_key,
        )
    raise ValueError("LEGAL_RAG_LLM_PROVIDER must be 'groq' or 'google'")


@contextmanager
def postgres_workflow(
    *,
    database_url: str | None = None,
    chat_model: Any | None = None,
    retrieve_fn: Any = retrieve_documents,
) -> Iterator[Any]:
    """Yield a graph with a PostgreSQL checkpointer and full checkpoint history."""

    database_url = database_url or settings.database_url
    if not database_url:
        raise RuntimeError("Set DATABASE_URL to enable PostgreSQL persistence")
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:
        raise RuntimeError(
            "Install langgraph-checkpoint-postgres and psycopg to persist conversations"
        ) from exc

    model = chat_model or create_chat_model()
    parts = urlsplit(database_url)
    if parts.scheme in {"postgres", "postgresql"}:
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.setdefault("connect_timeout", "5")
        database_url = urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
        )
    else:
        database_url = f"{database_url} connect_timeout=5"
    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        checkpointer.setup()
        yield build_legal_workflow(
            supervisor_model=model,
            domain_model=model,
            drafting_model=model,
            conversation_model=model,
            retrieve_fn=retrieve_fn,
            checkpointer=checkpointer,
        )
