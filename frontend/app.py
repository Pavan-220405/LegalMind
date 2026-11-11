"""Streamlit playground for the legal-assistance workflow."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, MutableMapping
from uuid import uuid4

import streamlit as st
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load local .env configuration for the LangSmith SDK, while preserving
# process environment and Streamlit secrets as higher-priority sources.
try:
    from dotenv import dotenv_values

    for _name, _value in dotenv_values(PROJECT_ROOT / ".env").items():
        if _value is not None:
            os.environ.setdefault(_name, _value)
except Exception:
    pass
for _name in (
    "LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT", "LANGSMITH_WORKSPACE_ID",
):
    try:
        _value = st.secrets.get(_name)
    except Exception:
        _value = None
    if _value is not None and str(_value).strip():
        os.environ[_name] = str(_value).strip()

if os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"true", "1", "yes", "on"}:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

LOGGER = logging.getLogger(__name__)


def get_or_create_thread_id(session_state: MutableMapping[str, Any]) -> str:
    """Keep one opaque thread ID for the lifetime of this Streamlit session."""

    thread_id = session_state.get("thread_id")
    if not thread_id:
        thread_id = uuid4().hex
        session_state["thread_id"] = thread_id
    return str(thread_id)


def start_new_conversation(session_state: MutableMapping[str, Any]) -> str:
    """Start a distinct checkpoint thread and clear session-only attachments."""

    thread_id = uuid4().hex
    session_state["thread_id"] = thread_id
    session_state.pop("attached_pdf_text", None)
    session_state.pop("attached_pdf_name", None)
    session_state.pop("last_used_pdf_hash", None)
    session_state["active_mode"] = "legal"
    return thread_id


def get_or_create_debate_thread_id(session_state: MutableMapping[str, Any]) -> str:
    thread_id = session_state.get("debate_thread_id")
    if not thread_id:
        thread_id = uuid4().hex
        session_state["debate_thread_id"] = thread_id
    return str(thread_id)


def start_debate_practice(session_state: MutableMapping[str, Any]) -> str:
    thread_id = uuid4().hex
    session_state["debate_thread_id"] = thread_id
    session_state["active_mode"] = "debate"
    session_state.pop("attached_pdf_text", None)
    session_state.pop("attached_pdf_name", None)
    session_state.pop("last_used_pdf_hash", None)
    return thread_id


def _get_memory_graph() -> Any:
    """Return this browser session's graph without Streamlit's global cache.

    The graph must live across Streamlit reruns because its MemorySaver stores
    conversation checkpoints. Keeping it in session state isolates it per user.
    """
    graph_key = "_legal_workflow_graph"
    if graph_key in st.session_state:
        return st.session_state[graph_key]

    from langgraph.checkpoint.memory import MemorySaver

    from agent.workflow import build_legal_workflow, create_chat_model

    model = create_chat_model()
    graph = build_legal_workflow(supervisor_model=model, checkpointer=MemorySaver())
    st.session_state[graph_key] = graph
    return graph


def _get_debate_graph() -> Any:
    """Return this browser session's debate graph across Streamlit reruns."""
    graph_key = "_debate_workflow_graph"
    if graph_key in st.session_state:
        return st.session_state[graph_key]

    from langgraph.checkpoint.memory import MemorySaver

    from agent.debate_agent import build_debate_workflow
    from agent.workflow import create_chat_model

    model = create_chat_model()
    graph = build_debate_workflow(
        supervisor_model=model,
        debate_model=model,
        checkpointer=MemorySaver(),
    )
    st.session_state[graph_key] = graph
    return graph


def _pdf_text(upload: Any) -> tuple[str, bool]:
    """Use the repository's mixed digital/scanned PDF extraction pipeline."""

    from agent.retriever.pdf_extractor import extract_document_text

    with tempfile.TemporaryDirectory(prefix="ai_lawyer_upload_") as temp_dir:
        path = Path(temp_dir) / "upload.pdf"
        path.write_bytes(upload.getvalue())
        return extract_document_text(path)


def _stream_turn_with_langsmith(
    graph: Any,
    *,
    thread_id: str,
    user_message: str,
    uploaded_pdf_text: str,
    turn_type: str,
    debate_mode: bool = False,
) -> Any:
    """Emit a LangSmith run for the workflow when tracing is configured."""

    if debate_mode:
        from agent.debate_agent.workflow import stream_debate_turn

        stream_fn = stream_debate_turn
    else:
        from agent.workflow import stream_turn

        stream_fn = stream_turn

    enabled = os.getenv("LANGSMITH_TRACING", "").strip().lower() in {
        "true", "1", "yes", "on"
    }
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not enabled or not api_key:
        yield from stream_fn(
            graph,
            thread_id=thread_id,
            user_message=user_message,
            turn_type=turn_type,
            **({} if debate_mode else {"uploaded_pdf_text": uploaded_pdf_text}),
        )
        return

    import langsmith as ls

    client = ls.Client(
        api_key=api_key,
        api_url=os.getenv("LANGSMITH_ENDPOINT", "").strip() or None,
    )
    project = os.getenv("LANGSMITH_PROJECT", "").strip() or None
    with ls.tracing_context(
        enabled=True,
        client=client,
        project_name=project,
        metadata={"thread_id": thread_id},
    ):
        yield from stream_fn(
            graph,
            thread_id=thread_id,
            user_message=user_message,
            turn_type=turn_type,
            **({} if debate_mode else {"uploaded_pdf_text": uploaded_pdf_text}),
        )


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
            else:
                pieces.append(str(item))
        return "\n".join(pieces)
    return str(content)


_STAGE_LABELS = {
    "record_user_turn": "Preparing conversation state",
    "safety_check": "Safety screening",
    "safety_response": "Returning safety guidance",
    "supervisor": "Supervisor — extracting facts and routing the issue",
    "rent_debate": "Rent & tenancy debate practice",
    "consumer_debate": "Consumer protection debate practice",
    "wages_debate": "Wages debate practice",
    "rent": "Rent retriever + specialist analysis",
    "consumer": "Consumer retriever + specialist analysis",
    "wages": "Wages retriever + specialist analysis",
    "general": "General legal-information fallback",
    "drafting": "Preparing the requested draft",
    "compliance": "Checking citations, urgency, and limitations",
    "conversation": "Composing the final response",
}


def _render_pipeline_trace(trace: dict[str, Any]) -> None:
    steps = trace.get("steps", [])
    if not steps and not trace.get("summary"):
        return
    with st.expander("Pipeline activity", expanded=False):
        for step in steps:
            icon = "✓" if step.get("status") == "completed" else "•"
            st.markdown(f"{icon} {step.get('label', 'Pipeline step')}")
        for item in trace.get("summary", []):
            st.caption(item)
        if trace.get("error"):
            st.caption(f"Stopped at: {trace['error']}")


def _trace_summary(result: dict[str, Any]) -> list[str]:
    summary = []
    safety_route = result.get("safety_route")
    if safety_route:
        summary.append(f"Safety route: {safety_route.replace('_', ' ')}")
    domain = result.get("domain")
    if domain:
        summary.append(f"Route: {domain}")
    sources = result.get("sources", [])
    if sources:
        summary.append(f"Retrieved sources: {len(sources)}")
    elif domain in {"rent", "consumer", "wages"}:
        summary.append("Retrieved sources: none")
    if result.get("drafted_document"):
        summary.append("Draft: prepared for review")
    if result.get("fallback_used"):
        summary.append("Response path: general fallback")
    if result.get("compliance_flag"):
        summary.append("Compliance review: one or more items need attention")
    return summary


def _render_history(graph: Any, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = graph.get_state(config)
        messages = snapshot.values.get("messages", []) if snapshot else []
    except Exception:
        messages = []

    traces_by_thread = st.session_state.get("pipeline_traces", {})
    traces = traces_by_thread.get(thread_id, [])
    traces_by_assistant_index = {
        trace.get("assistant_index"): trace
        for trace in traces
        if trace.get("assistant_index") is not None
    }
    assistant_index = 0
    for message in messages:
        role = "assistant" if getattr(message, "type", "") == "ai" else "user"
        with st.chat_message(role):
            st.markdown(_message_text(getattr(message, "content", message)))
            if role == "assistant":
                trace = traces_by_assistant_index.get(assistant_index)
                if trace is not None:
                    _render_pipeline_trace(trace)
                assistant_index += 1


def main() -> None:
    st.set_page_config(page_title="LegalMind AI", page_icon="⚖️", layout="wide")
    st.markdown("""
    <style>
      :root { --ink:#18243b; --muted:#627087; --accent:#b58a47; }
      .stApp { background: linear-gradient(145deg,#fbfaf7 0%,#f3f0e8 100%); color:var(--ink); }
      [data-testid="stSidebar"] { background:#f0ede5; border-right:1px solid #e2ddd1; }
      [data-testid="stSidebar"] > div:first-child { padding-top:2rem; }
      .brand-mark { color:var(--accent); font-size:1rem; letter-spacing:.24em; font-weight:700; }
      .hero { padding:2.1rem 0 1.2rem; border-bottom:1px solid #e4dfd4; margin-bottom:1.6rem; }
      .hero h1 { color:var(--ink); font-size:2.5rem; letter-spacing:-.045em; margin:.35rem 0 .4rem; }
      .hero p { color:var(--muted); font-size:1.03rem; margin:0; }
      [data-testid="stChatMessage"] { border-radius:16px; border:1px solid #ebe6dc; box-shadow:0 5px 20px #29334a0a; }
      [data-testid="stChatInput"] { border-color:#d8d1c3; border-radius:14px; }
      .stButton > button { border-radius:10px; border-color:#d9d2c5; color:var(--ink); }
      .stButton > button[kind="primary"] { background:#1c2941; border-color:#1c2941; color:white; }
      [data-testid="stFileUploader"] { background:#fffefa; border:1px dashed #c9b78f; border-radius:12px; }
      .debate-panel { background:#e6eff0; border:1px solid #c8dadd; border-left:5px solid #547d84; border-radius:14px; padding:1rem 1.2rem; margin:0 0 1.2rem; }
      .debate-panel .eyebrow { color:#547d84; font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; margin-bottom:.35rem; }
      .debate-panel h2 { color:#203c45; font-size:1.45rem; margin:0 0 .35rem; }
      .debate-panel p { color:#405961; margin:0; }
    </style>
    <div class="hero"><div class="brand-mark">LEGALMIND · AI</div>
      <h1>LegalMind AI</h1>
      <p>Clear, practical legal guidance, grounded in your documents.</p></div>
    """, unsafe_allow_html=True)

    if "thread_id" not in st.session_state:
        get_or_create_thread_id(st.session_state)
    st.session_state.setdefault("active_mode", "legal")

    with st.sidebar:
        st.markdown('<div class="brand-mark">LEGALMIND · AI</div>', unsafe_allow_html=True)
        st.caption("Your private legal workspace")
        with st.expander("LangSmith tracing", expanded=False):
            tracing_enabled = os.getenv("LANGSMITH_TRACING", "").strip().lower() in {
                "true", "1", "yes", "on"
            }
            has_api_key = bool(os.getenv("LANGSMITH_API_KEY", "").strip())
            project = os.getenv("LANGSMITH_PROJECT", "").strip()
            if tracing_enabled and has_api_key:
                st.success("Tracing is enabled")
                st.caption(f"Project: {project or 'default'}")
            elif tracing_enabled:
                st.warning("Tracing is enabled, but LANGSMITH_API_KEY is missing.")
            else:
                st.caption("Set LANGSMITH_TRACING=true and configure LANGSMITH_API_KEY to enable.")
        if st.button("New conversation", use_container_width=True):
            start_new_conversation(st.session_state)
            st.rerun()
        if st.button(
            "Debate practice",
            use_container_width=True,
            type="primary" if st.session_state.get("active_mode") == "debate" else "secondary",
        ):
            start_debate_practice(st.session_state)
            st.rerun()
        st.caption(
            "Practice is available for rent and tenancy, consumer protection, "
            "and wages. More legal areas and practice features are coming."
        )

        debate_active = st.session_state.get("active_mode") == "debate"
        upload = st.file_uploader(
            "Add a PDF",
            type=["pdf"],
            key="pdf_upload",
            disabled=debate_active,
            help=(
                "Debate Practice uses the legal knowledge base."
                if debate_active
                else "PDF files up to 10 MB"
            ),
        )
        if debate_active:
            upload = None
            st.session_state.pop("last_used_pdf_hash", None)
        if upload is None:
            st.session_state.pop("last_used_pdf_hash", None)
        if upload is not None:
            upload_hash = hashlib.sha256(upload.getvalue()).hexdigest()
            if st.session_state.get("last_used_pdf_hash") != upload_hash:
                try:
                    with st.spinner("Extracting PDF text…"):
                        text, used_ocr = _pdf_text(upload)
                    if not text.strip():
                        st.warning("No text could be extracted from this PDF.")
                    else:
                        st.session_state["attached_pdf_text"] = text
                        st.session_state["attached_pdf_name"] = upload.name
                        st.session_state["last_used_pdf_hash"] = upload_hash
                        st.success(
                            f"Attached {upload.name} "
                            f"({'OCR used' if used_ocr else 'text extracted'})."
                        )
                except Exception as exc:
                    st.error(f"PDF extraction failed ({type(exc).__name__}).")
        if st.session_state.get("attached_pdf_text"):
            st.caption(f"PDF ready: {st.session_state['attached_pdf_name']}")

    debate_mode = st.session_state.get("active_mode") == "debate"
    thread_id = (
        get_or_create_debate_thread_id(st.session_state)
        if debate_mode
        else get_or_create_thread_id(st.session_state)
    )
    try:
        with st.spinner("Preparing debate practice…" if debate_mode else "Preparing your legal workspace…"):
            graph = _get_debate_graph() if debate_mode else _get_memory_graph()
    except Exception as exc:
        st.error(f"Could not initialize the workflow ({type(exc).__name__}).")
        st.stop()

    if debate_mode:
        st.markdown(
            """
            <section class="debate-panel">
              <div class="eyebrow">Practice mode</div>
              <h2>Debate Practice</h2>
              <p>You’ll argue your chosen position. The AI will take the opposing side using retrieved legal sources. It will clarify the setup before debate begins. Type “coach me” for guidance.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )

    _render_history(graph, thread_id)

    prompt = st.chat_input(
        "Describe the issue and your side to practise"
        if debate_mode
        else "Describe the issue or ask a follow-up"
    )
    if prompt:
        st.chat_message("user").markdown(prompt)
        pdf_text = st.session_state.get("attached_pdf_text", "")
        steps: list[dict[str, str]] = []
        trace: dict[str, Any] = {"steps": steps, "summary": []}
        status = None
        prior_assistant_count = 0
        try:
            with st.chat_message("assistant"):
                with st.status("Starting the pipeline…", expanded=True) as status:
                    activity = st.empty()
                    snapshot = graph.get_state(
                        {"configurable": {"thread_id": thread_id}}
                    )
                    prior_messages = (
                        snapshot.values.get("messages", []) if snapshot else []
                    )
                    prior_assistant_count = sum(
                        getattr(message, "type", "") == "ai"
                        for message in prior_messages
                    )
                    turn_type = "followup" if prior_messages else "new_problem"
                    for event in _stream_turn_with_langsmith(
                        graph,
                        thread_id=thread_id,
                        user_message=prompt,
                        uploaded_pdf_text=pdf_text,
                        turn_type=turn_type,
                        debate_mode=debate_mode,
                    ):
                        if not isinstance(event, dict):
                            continue
                        event_type = event.get("type")
                        payload = event.get("payload", {})
                        node_name = payload.get("name")
                        if not node_name:
                            continue
                        label = _STAGE_LABELS.get(node_name, node_name.replace("_", " ").title())
                        if event_type == "task":
                            steps.append({"label": label, "status": "running"})
                            status.update(label=f"Running: {label}", state="running")
                            activity.markdown(
                                "\n".join(
                                    f"• {step['label']}" for step in steps
                                )
                            )
                        elif event_type == "task_result":
                            for step in reversed(steps):
                                if step["label"] == label and step["status"] == "running":
                                    step["status"] = "completed"
                                    break
                            activity.markdown(
                                "\n".join(
                                    f"{'✓' if step['status'] == 'completed' else '•'} "
                                    f"{step['label']}"
                                    for step in steps
                                )
                            )
                            if payload.get("error") is not None:
                                raise payload["error"]

                    # LangChain sends traces on a background queue. Wait for the
                    # queue so short-lived hosted workers can flush each turn.
                    try:
                        from langchain_core.tracers.langchain import wait_for_all_tracers

                        wait_for_all_tracers()
                    except Exception:
                        LOGGER.exception("Could not flush LangSmith trace queue")

                    snapshot = graph.get_state(
                        {"configurable": {"thread_id": thread_id}}
                    )
                    result = snapshot.values if snapshot else {}
                    trace["summary"] = _trace_summary(result)
                    status.update(label="Pipeline complete", state="complete", expanded=False)
                    st.markdown(_message_text(result.get("final_response", "")))
                    _render_pipeline_trace(trace)

            st.session_state.setdefault("pipeline_traces", {}).setdefault(
                thread_id, []
            ).append({**trace, "assistant_index": prior_assistant_count})
            if len(st.session_state["pipeline_traces"][thread_id]) > 24:
                st.session_state["pipeline_traces"][thread_id] = st.session_state[
                    "pipeline_traces"
                ][thread_id][-24:]
            if pdf_text:
                st.session_state.pop("attached_pdf_text", None)
                st.session_state.pop("attached_pdf_name", None)
            st.rerun()
        except Exception as exc:
            LOGGER.exception("Workflow turn failed")
            failed_node = next(
                (
                    step["label"]
                    for step in reversed(steps)
                    if step["status"] == "running"
                ),
                "the pipeline",
            )
            trace["error"] = failed_node
            if type(exc).__name__ == "NotFoundError":
                st.error(
                    "The configured model or retrieval endpoint was not found. "
                    "Restart Streamlit to load the updated model defaults. If the "
                    "error remains, check LEGAL_MODEL, SMALL_MODEL, and reranker settings."
                )
                st.stop()
            st.error(
                f"Pipeline failed during {failed_node} ({type(exc).__name__}). "
                "Check the app logs for details."
            )
            st.stop()

    with st.sidebar:
        st.divider()
        st.subheader("Knowledge base")
        try:
            from agent.retriever.chroma import collection_count
            from agent.retriever.config import COLLECTION_NAMES

            counts = {
                domain: collection_count(name)
                for domain, name in COLLECTION_NAMES.items()
            }
            for domain, count in counts.items():
                st.write(f"{domain.title()}: {count:,} chunks")
            if not any(counts.values()):
                st.warning("No legal chunks found. Run `python -m agent.retriever.ingest` first.")
        except Exception as exc:
            st.warning(f"Could not read Chroma status ({type(exc).__name__}).")


if __name__ == "__main__":
    main()
