"""Streamlit playground for the legal-assistance workflow."""

from __future__ import annotations

import atexit
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

# Load local .env values for LangSmith's own SDK (which reads os.environ), then
# let process variables and Streamlit secrets take precedence. Importing config
# constructs its Pydantic Settings singleton immediately, so doing this afterward
# would leave the singleton with stale .env values for the life of the process.
try:
    from dotenv import dotenv_values

    for _secret_name, _secret_value in dotenv_values(PROJECT_ROOT / ".env").items():
        if _secret_value is not None:
            os.environ.setdefault(_secret_name, _secret_value)
except Exception:
    pass
for _secret_name in (
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT",
    "LANGSMITH_WORKSPACE_ID",
):
    try:
        _secret_value = st.secrets.get(_secret_name)
    except Exception:
        _secret_value = None
    if _secret_value is not None and str(_secret_value).strip():
        os.environ[_secret_name] = str(_secret_value).strip()
if os.getenv("LANGSMITH_TRACING"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.environ["LANGSMITH_TRACING"]

from config import settings

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
    return thread_id


def get_thread_id_for_mode(
    session_state: MutableMapping[str, Any], mode: str
) -> str:
    """Keep one thread per session and checkpoint backend."""

    previous_mode = session_state.get("checkpoint_mode")
    if previous_mode is not None and previous_mode != mode:
        start_new_conversation(session_state)
        session_state["checkpoint_mode_changed"] = True
    session_state["checkpoint_mode"] = mode
    return get_or_create_thread_id(session_state)


@st.cache_resource(show_spinner=False)
def _get_memory_graph() -> Any:
    from langgraph.checkpoint.memory import MemorySaver

    from agent.workflow import build_legal_workflow, create_chat_model

    model = create_chat_model()
    return build_legal_workflow(supervisor_model=model, checkpointer=MemorySaver())


@st.cache_resource(show_spinner=False)
def _get_postgres_graph(database_url: str) -> Any:
    from agent.workflow import create_chat_model, postgres_workflow

    model = create_chat_model()
    workflow_context = postgres_workflow(database_url=database_url, chat_model=model)
    graph = workflow_context.__enter__()
    atexit.register(workflow_context.__exit__, None, None, None)
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
) -> Any:
    """Run the graph under an explicit LangSmith client when tracing is enabled."""

    from agent.workflow import stream_turn

    enabled = os.getenv("LANGSMITH_TRACING", "").strip().lower() in {
        "true", "1", "yes", "on"
    }
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not enabled or not api_key:
        yield from stream_turn(
            graph,
            thread_id=thread_id,
            user_message=user_message,
            uploaded_pdf_text=uploaded_pdf_text,
            turn_type=turn_type,
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
        yield from stream_turn(
            graph,
            thread_id=thread_id,
            user_message=user_message,
            uploaded_pdf_text=uploaded_pdf_text,
            turn_type=turn_type,
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
    st.set_page_config(page_title="AI Lawyer Pipeline", page_icon="⚖️", layout="wide")
    st.title("AI Lawyer — pipeline playground")
    st.caption(
        "Try intake, domain routing, retrieval, drafting, compliance, and follow-up turns. "
        "Pipeline activity is shown under each response."
    )

    with st.sidebar:
        with st.expander("LangSmith tracing diagnostics", expanded=False):
            tracing_enabled = os.getenv("LANGSMITH_TRACING", "").strip().lower() in {
                "true", "1", "yes", "on"
            }
            api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
            endpoint = os.getenv("LANGSMITH_ENDPOINT", "").strip()
            project = os.getenv("LANGSMITH_PROJECT", "").strip()
            st.write(f"Tracing enabled: {'yes' if tracing_enabled else 'no'}")
            st.write(f"API key configured: {'yes' if api_key else 'no'}")
            st.write(f"Endpoint configured: {'yes' if endpoint else 'no (US default)'}")
            st.write(f"Project: {project or '(default project)'}")
            if api_key:
                if st.button("Verify LangSmith connection", key="verify_langsmith"):
                    try:
                        from langsmith import Client

                        client = Client(api_url=endpoint or None, api_key=api_key)
                        # The SDK's info property makes a request and validates
                        # authentication, endpoint reachability, and workspace.
                        _ = client.info
                        project_found = True
                        if project:
                            projects = list(client.list_projects(name=project, limit=1))
                            project_found = bool(projects)
                    except Exception as exc:
                        st.error(
                            "LangSmith verification failed "
                            f"({type(exc).__name__}). Check the key, workspace/region, "
                            "and Streamlit Cloud logs; the secret value is not shown."
                        )
                    else:
                        st.success("LangSmith accepted the configured credentials.")
                        if project and not project_found:
                            st.info(
                                "That project does not exist yet. Run one app query; "
                                "LangSmith should create it when the first trace arrives."
                            )

    if "thread_id" not in st.session_state:
        get_or_create_thread_id(st.session_state)

    with st.sidebar:
        st.header("Conversation")
        mode = st.radio(
            "Checkpoint storage",
            ("PostgreSQL (persistent)", "In-memory demo (temporary)"),
            index=1,
            key="checkpoint_mode_widget",
            help=(
                "In-memory mode is ready for a local pipeline check and clears when "
                "the process stops. PostgreSQL keeps threads across restarts when its "
                "database exists and is reachable."
            ),
        )
        thread_id = get_thread_id_for_mode(st.session_state, mode)
        if st.session_state.pop("checkpoint_mode_changed", False):
            st.info("Storage mode changed, so a new conversation thread was started.")
        st.caption("Thread ID (managed by this Streamlit session)")
        st.code(thread_id, language=None)
        if st.button("New conversation", use_container_width=True):
            start_new_conversation(st.session_state)
            st.rerun()

        st.divider()
        st.subheader("Attach a PDF")
        upload = st.file_uploader("PDF evidence", type=["pdf"], key="pdf_upload")
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
            st.caption(f"Ready for the next message: {st.session_state['attached_pdf_name']}")
            if st.button("Remove attached PDF", use_container_width=True):
                st.session_state.pop("attached_pdf_text", None)
                st.session_state.pop("attached_pdf_name", None)
                if upload is not None:
                    st.session_state["last_used_pdf_hash"] = upload_hash
                else:
                    st.session_state.pop("last_used_pdf_hash", None)
                st.rerun()
            if upload is not None and st.button(
                "Use this PDF again", use_container_width=True
            ):
                st.session_state.pop("last_used_pdf_hash", None)
                st.rerun()
        elif (
            upload is not None
            and st.session_state.get("last_used_pdf_hash") == upload_hash
            and st.button("Attach selected PDF again", use_container_width=True)
        ):
            st.session_state.pop("last_used_pdf_hash", None)
            st.rerun()

    if mode == "PostgreSQL (persistent)":
        database_url = (settings.database_url or "").strip()
        if not database_url:
            st.error("Set DATABASE_URL in the project .env file before using PostgreSQL mode.")
            st.stop()
        try:
            with st.spinner("Opening the PostgreSQL workflow…"):
                graph = _get_postgres_graph(database_url)
        except Exception as exc:
            st.error(
                "Could not open PostgreSQL workflow "
                f"({type(exc).__name__}). Confirm the database named in DATABASE_URL "
                "exists and the server is running. You can use in-memory demo mode meanwhile."
            )
            st.stop()
    else:
        try:
            with st.spinner("Loading the model and workflow…"):
                graph = _get_memory_graph()
        except Exception as exc:
            st.error(f"Could not initialize the workflow ({type(exc).__name__}).")
            st.stop()

    thread_id = get_thread_id_for_mode(st.session_state, mode)
    _render_history(graph, thread_id)

    prompt = st.chat_input("Describe the issue or ask a follow-up")
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

                    # LangChain sends traces on a background queue. Drain it
                    # before Streamlit completes the request so short-lived
                    # hosted workers don't drop the final trace batch.
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
