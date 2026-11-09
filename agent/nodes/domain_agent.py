"""Shared implementation for retrieval-grounded domain agents."""

from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from agent.prompts.domain import DOMAIN_ANALYSIS_PROMPT
from agent.nodes.structured import bind_structured_output
from agent.schemas.domain import SubAgentOutput
from agent.schemas.state import BaseState

RetrieveFn = Callable[..., list[Document]]


def _query_for(state: BaseState) -> str:
    return "\n".join(
        part
        for part in (
            state.get("problem_statement", ""),
            state.get("user_message", ""),
            "Timeline as stated: " + state.get("timeline_raw", ""),
        )
        if part.strip()
    )


def _source_record(source_id: str, document: Document) -> dict[str, object]:
    metadata = document.metadata or {}
    title = str(metadata.get("document_name") or metadata.get("source_path") or "Legal source")
    return {
        "source_id": source_id,
        "title": title,
        "section_number": str(metadata.get("section_number") or ""),
        "section_title": str(metadata.get("section_title") or ""),
        "page_start": metadata.get("page_start"),
        "page_end": metadata.get("page_end"),
        "source_path": str(metadata.get("source_path") or ""),
    }


def build_domain_agent(
    domain: str,
    collection: str,
    chat_model: Any,
    retrieve_fn: RetrieveFn,
) -> Callable[[BaseState], dict[str, object]]:
    """Build one complete domain path: retrieval followed by grounded analysis."""

    structured_model = bind_structured_output(chat_model, SubAgentOutput)

    def domain_agent(state: BaseState) -> dict[str, object]:
        query = _query_for(state)
        documents = retrieve_fn(query, collection) if query else []
        source_records = [
            _source_record(f"S{index}", document)
            for index, document in enumerate(documents, start=1)
        ]
        excerpts = "\n\n".join(
            f"[{record['source_id']}] {record['title']}"
            f"; section {record['section_number']}; pages "
            f"{record['page_start']}-{record['page_end']}\n{document.page_content}"
            for record, document in zip(source_records, documents)
        ) or "No legal excerpts were retrieved. Do not state collection-specific law."
        facts = (
            f"Issue: {state.get('problem_statement', '')}\n"
            f"Timeline raw: {state.get('timeline_raw', '')}\n"
            f"Timeline parsed (explicit only): {state.get('timeline_parsed', '')}\n"
            f"Evidence stated by user: {state.get('proofs_stated', [])}\n"
            f"Evidence extracted from PDF: {state.get('proofs_from_pdf', [])}\n"
            f"Extraction confidence: {state.get('extraction_confidence', 'low')}\n"
            f"Urgency: {state.get('urgency', 'normal')}"
        )
        response = structured_model.invoke(
            [
                SystemMessage(content=DOMAIN_ANALYSIS_PROMPT.format(domain=domain)),
                HumanMessage(content=f"Facts:\n{facts}\n\nLegal excerpts:\n{excerpts}"),
            ]
        )
        output = (
            response
            if isinstance(response, SubAgentOutput)
            else SubAgentOutput.model_validate(response)
        )
        valid_ids = {record["source_id"] for record in source_records}
        invalid_citations = [
            citation.source_id
            for citation in output.citations
            if citation.source_id not in valid_ids
        ]
        output.citations = [
            citation for citation in output.citations if citation.source_id in valid_ids
        ]
        return {
            "subagent_output": output.model_dump(),
            "sources": source_records,
            "citation_audit_errors": invalid_citations,
        }

    return domain_agent
