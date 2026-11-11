from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

PATH = r"F:\AI_Lawyer\output\docx\legalmind-solution-report-academic.docx"
doc = Document(PATH)

# Correct over-specific configuration statements in the original report.
for p in doc.paragraphs:
    if p.text.startswith("Cohere is the reranker used by the project configuration"):
        p.text = ("The repository exposes Cohere and Voyage reranker integrations. "
                  "The checked-in Settings defaults name Voyage rerank-3-lite, while "
                  "README examples describe Cohere rerank-v5.0-fast; environment "
                  "configuration can override both. The exact active provider and "
                  "model therefore depend on the runtime .env or process environment. "
                  "Reranking compares candidates with the original question, reorders "
                  "them, and returns up to six documents by default. It cannot verify "
                  "current legal status, jurisdictional fit, or legal entailment.")
    if p.text.startswith("Keywords:") and "Cohere reranking" in p.text:
        p.text = p.text.replace("Cohere reranking", "cross-encoder reranking")

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)

def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement('w:tblHeader')
    tblHeader.set(qn('w:val'), 'true')
    trPr.append(tblHeader)

def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Light Shading Accent 1'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    repeat_header(t.rows[0])
    for i, h in enumerate(headers):
        c=t.rows[0].cells[i]; c.text=h; shade(c,'1F3B5B')
        for r in c.paragraphs[0].runs:
            r.font.bold=True; r.font.color.rgb=__import__('docx').shared.RGBColor(255,255,255)
    for vals in rows:
        cells=t.add_row().cells
        for i,v in enumerate(vals):
            cells[i].text=str(v)
            cells[i].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    doc.add_paragraph()
    return t

def heading(text, level=1):
    doc.add_heading(text, level=level)

def para(text):
    return doc.add_paragraph(text)

def bullet(text):
    doc.add_paragraph(text, style='List Bullet')

def numbered(text):
    doc.add_paragraph(text, style='List Number')

doc.add_page_break()
heading('9. Detailed system specification', 1)
para('This section expands the implementation map into the observable data flow and responsibilities of each subsystem. The descriptions refer to code present in the repository at the time this report was prepared. Configuration values may be overridden by environment variables; secrets are intentionally not reproduced here.')
heading('9.1 Request lifecycle in Legal Help', 2)
table(['Stage','Implemented behavior','Primary state or output'], [
('Turn recording','Adds the user turn to LangGraph message history and resets turn-specific analysis fields while preserving conversation context.','thread_id, messages, user_message'),
('Safety screen','Local deterministic pattern checks run before model routing for explicit self-harm or imminent violence language. Matched cases return a direct support response without legal-agent calls.','safety_route, final_response'),
('Supervisor intake','Structured model output extracts issue, domain, confidence, explicit dates, evidence, and urgency from the current turn, prior context, and optional uploaded PDF text. Low confidence or uncertain routing asks for clarification.','BaseState fields; clarification_needed'),
('Domain retrieval and analysis','The rent, consumer, or wages agent queries only its mapped collection, attaches source IDs and metadata, and asks a structured model to reason from returned excerpts. General issues use a caveated fallback.','subagent_output, sources, citation_audit_errors'),
('Drafting','A separate model step prepares a requested notice or communication from stated facts and verified legal information; missing details are bracketed placeholders.','drafted_document'),
('Compliance checks','A deterministic phase-one check audits citation IDs, uncited legal information, limitation uncertainty, and urgency.','compliance_flag, compliance_notes, limitation_status'),
('Section correction','Unsupported section-like references in key answer fields and drafts are changed to the single unambiguous retrieved section, or have the number removed when sources are ambiguous.','section_correction_notes'),
('Conversation response','The final node presents a user-facing answer using the accumulated state and checkpointed conversation.','final_response, messages'),
])
para('The graph has one route per supported domain plus general fallback, clarification, and safety branches. Domain agents converge on drafting, compliance, section correction, and final conversation. Provider model objects, retrieval functions, and checkpointers are injectable, allowing tests to substitute fakes and local memory without production API calls.')
heading('9.2 Structured state and conversation continuity', 2)
para('The shared BaseState carries user_message, thread_id, turn_type, messages, the extracted problem statement and domain, domain and extraction confidence, raw and explicitly parsed timeline values, evidence stated by the user, evidence extracted from an uploaded PDF, urgency, retrieved sources, structured subagent output, draft, compliance notes, and final response. LangGraph message reducers append conversation messages; the implementation does not maintain a second manually synchronized history list. The supervisor sees recent prior turns, while the domain step receives the structured facts and evidence references. Relative dates remain raw until explicitly parsed, so “three months ago” is not silently converted into a calendar date.')
para('The debate graph uses its own DebateState for practice topic, domain, position, jurisdiction, readiness, coaching request, retrieved contexts, and the current argument. A first turn gathers and confirms the setup. A later turn starts the retrieval-backed opposing argument. Coaching uses excerpts cached from the preceding debate turn. This separation keeps the practice activity distinct from a real legal-help response.')
heading('9.3 Retrieval algorithm and tuning points', 2)
numbered('Canonicalize the requested domain alias and map it to one configured Chroma collection. Public aliases include rent/property, wages/labour, and consumer.')
numbered('Optionally create three alternative queries with the configured small model. Reject duplicates or invalid rewrites and preserve factual tokens (section references, years, dates, amounts, and durations) from being omitted.')
numbered('For each query, obtain dense Chroma candidates and BM25 candidates reconstructed from persisted chunk records. The configured default candidate counts are 20 each.')
numbered('Combine rankings using LangChain EnsembleRetriever weighted reciprocal-rank fusion. Default weights are 0.5 dense and 0.5 lexical; weights and retrieval count are configurable.')
numbered('Merge expanded-query results by stable chunk identifier, then optionally rerank against the original user query. The default final display count is six.')
para('The retrieval module returns LangChain Document objects and does not itself generate legal answers. BM25 is rebuilt lazily after process restart and cached per collection; ingestion invalidates that cache when the source changes. Deterministic chunk IDs incorporate collection, path, source hash, order, page range, embedding marker, and content. These mechanisms aid reproducibility and incremental updates, but neither guarantees identical ranked results across model, library, corpus, or configuration changes.')
table(['Setting','Default in Settings','Operational meaning'], [
('VECTOR_DB / CHROMA_PERSIST_DIRECTORY','chroma / ./data/chroma','Local persistent vector store and storage path.'),
('DENSE_CANDIDATES / BM25_CANDIDATES','20 / 20','Candidate pool sizes for semantic and lexical search.'),
('DENSE_WEIGHT / BM25_WEIGHT','0.5 / 0.5','Fusion weights for the two ranked lists.'),
('MULTI_QUERY_ENABLED / MULTI_QUERY_COUNT','true / 3','Query expansion switch and number of alternatives.'),
('RERANKER_ENABLED / RERANK_TOP_K','true / 6','Reranker switch and final candidate count.'),
('HF_EMBEDDING_MODEL / DEVICE','BAAI/bge-m3 / cpu','Embedding model and configured inference device.'),
('LEGAL_RAG_LLM_PROVIDER','auto groq or google','Provider selection; provider keys and model names are environment-backed.'),
('CONVERSATION_CONTEXT_MESSAGES','16','Configured maximum conversation context setting.'),
])
para('The repository has a configuration discrepancy worth resolving before a reproducible deployment: config.py defaults to Voyage and rerank-3-lite, while README examples describe Cohere and rerank-v5.0-fast. The settings model accepts either provider and environment overrides. Operators should record the effective settings (excluding secrets) with each evaluation run.')

heading('10. Legal corpus and data preparation', 1)
para('The current source corpus is a compact, manually enumerated set of nine PDF files. Domain assignment is explicit in config.py, not an unconstrained runtime classifier. One Social Security Code PDF resides in the consumer folder on disk but is deliberately assigned to the wages collection. Sources are not automatically refreshed from government websites, and corpus inclusion does not establish that a statute has commenced, remains current, or applies in a particular state.')
table(['Collection / user lane','Configured sources','Coverage boundary'], [
('consumer_law / Consumer protection','Consumer Protection Act, 2019; Consumer Protection (General) Rules, 2020; Consumer Protection (Consumer Commission) Regulations, 2020.','Central consumer statute and selected rules/regulations. No exhaustive case law, state practice, or complete current forum guidance.'),
('wages_law / Wages and selected social security','Code on Wages, 2019; Code on Wages (Amendment), 2026; Code on Social Security, 2020.','Selected wage and social-security texts. Applicable notifications, commencement, schemes, and worker-specific status require verification.'),
('rent_law / Rent and tenancy','Model Tenancy Act, 2021; Registration Act, 1908; Transfer of Property Act, 1882.','Central/model and general property texts; the Model Tenancy Act is a model framework and state adoption/local rules matter.'),
])
heading('10.1 PDF extraction, chunking, and metadata', 2)
para('Digital PDF text is extracted page by page. Scanned or low-text pages can invoke OCR through OCRmyPDF/Tesseract, with configured languages English and Hindi; OCR command, Tesseract location, and language are configurable. Conservative cleanup is applied after extraction. The pipeline detects chapters, parts, sections, rules, regulations, and article-like headings, and retains page spans. It divides long legal provisions by subsections and paragraphs; oversized text is split into word windows with overlap. The default target is roughly 650 estimated tokens per chunk, with a 40-word overlap for oversized windows. If no statutory boundaries are detected, paragraph-based chunks are used.')
para('Each stored chunk can carry: collection and domain; document title, type, and year; relative source path and filename; source SHA-256; start/end page; detected section number and title; chapter; subsection; chunk type and index; chunk identifier; and embedding-model marker. This provenance enables display, source-grounded prompts, incremental ingestion, and evaluation joins. Parsing errors in a scanned statute can still mislabel boundaries or pages and require source inspection.')
heading('10.2 Corpus governance required for production', 2)
bullet('Maintain a source register with issuing authority, official URL or publication record, download date, checksum, version, amendment status, commencement status, and jurisdictions covered.')
bullet('Add state and local materials only after expert review and explicit effective-date and jurisdiction metadata design.')
bullet('Re-ingest changed PDFs, validate extraction against sampled page images, verify detected sections, and record the resulting chunk-count and source-hash change log.')
bullet('Define a deprecation and rollback process for superseded or withdrawn source files; do not present a PDF upload as authoritative merely because it was ingested.')

heading('11. User-facing functionality and boundaries', 1)
table(['Capability','Current behavior','Boundary'], [
('Legal Help','Text conversation, new-conversation control, domain routing, retrieved citations, and optional requested communication draft.','Information support only; no representation, filing, sending, or guaranteed outcome.'),
('PDF attachment','A user can attach a PDF up to 10 MB; text is extracted into a temporary directory and supplied to intake for the turn.','PDF text is user-provided context, not verified law. OCR and extraction can fail or omit content.'),
('Debate Practice','Rent/tenancy, consumer, or wages; setup confirmation; source-labelled opposing argument; follow-up turn; coaching on request.','Practice only. It does not predict a judge or provide a complete legal strategy.'),
('Pipeline activity','The interface streams task lifecycle events and can show a high-level node trace.','Trace summaries are operational signals, not legal-quality scores.'),
('General fallback','For unsupported issues, provides brief general information and asks for jurisdiction or missing facts when useful.','No verified domain collection is attached; jurisdiction-specific propositions are constrained.'),
('Safety routing','Recognized immediate-harm patterns bypass legal model routing and return emergency/support guidance.','Pattern matching has false negatives and false positives; it is not a clinical assessment.'),
])
para('The Streamlit application provides Legal Help and Debate Practice modes and uses in-memory checkpointing by default. A helper can build the normal legal graph with PostgreSQL checkpointing; the debate graph defaults to memory. In the UI, process shutdown clears in-memory state. A new conversation generates a distinct opaque thread ID and clears temporary attachment state. PDF uploads are not automatically ingested into the legal collections.')

heading('12. Verification and evaluation evidence', 1)
heading('12.1 What has been specified', 2)
para('The repository contains test modules for retrieval, Chroma persistence, ingestion metadata, rent queries, debate workflow, and integration behavior. Manual retrieval fixtures cover twelve queries across the three collections, compare dense, BM25, and hybrid behavior, and inspect metadata, deduplication, toggles, and collection isolation. The fifteen-case legal QA suite covers evidence preservation, jurisdiction ambiguity, multi-turn continuity, consumer standing, product remedies, limitation, wage deductions and delay, gender pay, social-security classification, and a cross-domain scope case. A debate scenario checks intake, confirmation, rent routing, excerpt-grounded opposition, continuity, coaching, and no-source abstention.')
para('The QA documents are test specifications and reference targets, not results. They explicitly instruct evaluators to inspect the selected collection, source title/path, returned chunk IDs, page and section metadata, final answer, and uncertainty. RAGAS is proposed for faithfulness, answer relevance, context precision/recall, and factual correctness where supported; real retrieved excerpts must be used as reference contexts rather than prose labels.')
heading('12.2 Evidence available and missing', 2)
table(['Evidence item','Status in repository','What it supports'], [
('Automated unit/integration tests','Test files are present; this report does not claim a fresh test run.','Test coverage exists for graph behavior and retrieval components; passing status must be captured from a dated run.'),
('Manual case suites','Retrieval, QA, debate, and anti-hallucination cases are documented.','A repeatable evaluation protocol, not measured accuracy.'),
('Anti-hallucination probes','Three deterministic behavior branches are documented; one relative-date extraction case is model-driven. Provider-backed live transcripts were not captured due to provider connectivity failure.','Evidence for deterministic branch outputs only; not evidence of live model adherence.'),
('evals/main_agent_runs.csv','Template rows are empty; no answer, context, model, or review values are populated.','No scored answer-level evaluation results are available in this file.'),
('Latency, cost, reliability','No benchmark table or run log is supplied.','No quantitative deployment or service-level claim can be made.'),
('Independent legal review','No signed review or expert-labeled benchmark is supplied.','Legal correctness and completeness remain unvalidated.'),
])
para('Therefore, the project can claim implemented workflows and evaluation fixtures, but it cannot claim a measured accuracy rate, verified hallucination reduction, legally validated answers, or production readiness on the evidence currently present. A future results section should report sample size, date, exact code revision, corpus version, effective model/provider, retrieval settings, annotator qualifications, rubric, per-domain results, uncertainty intervals, and failure examples.')
heading('12.3 Recommended evaluation protocol', 2)
numbered('Freeze a code revision, corpus manifest, effective settings, and model versions. Remove all credentials from the record.')
numbered('Run deterministic unit tests and separately record provider-backed tests, indicating which were skipped, blocked, or failed.')
numbered('For every case, save the prompt and relevant prior turns, actual retrieved chunk IDs and text, final response, model identifier, collection, latency, estimated provider cost, and adjudication notes.')
numbered('Have at least two qualified reviewers grade legal claim support, citation entailment, jurisdiction and date handling, material omission, calibration, safety, and usefulness; resolve disagreements and report agreement.')
numbered('Report retrieval metrics separately from answer metrics. Include abstention correctness and negative/no-source cases. Do not collapse safety, clarification, and legal answer behavior into one score.')
numbered('Retest regressions after source, prompt, model, dependency, embedding, or reranker changes; retain representative failures and remediation decisions.')

heading('13. Security, privacy, and operational controls', 1)
para('The application processes highly sensitive narratives, employment and tenancy records, and uploaded legal documents. The current implementation uses temporary local storage for PDF extraction and optional external model providers. Optional LangSmith tracing can transmit workflow inputs, outputs, metadata, retrieved excerpts, or uploaded text to the configured tracing service. The repository itself cautions that traces may contain legal narratives and source text.')
table(['Area','Current implementation or exposure','Control needed before broad deployment'], [
('Secrets','Provider and database keys are read from environment-backed settings; secret values are not intended for reports or logs.','Use a secret manager, rotate credentials, constrain scopes, and prevent secret values from entering traces or diagnostics.'),
('Uploaded PDFs','Copied to a temporary directory for extraction; extracted text may enter prompts and checkpoints.','Define retention/deletion, size and parser limits, malware and malformed-file handling, and clear notice of model processing.'),
('Conversation state','Streamlit memory by default; PostgreSQL checkpointer is available for normal workflow.','Encrypt storage and backups, isolate tenant/thread access, set expiry and deletion controls, and test authorization boundaries.'),
('Tracing','Optional external tracing controlled by configuration.','Default off for sensitive use unless governed; redact/minimize inputs, restrict access, configure retention and region, and inform users.'),
('Local Chroma','Persistent vector data under configured local directory.','Protect filesystem permissions and backups; document deletion and reindex procedures.'),
('Access and deployment','Repository is a development prototype; no authentication, multi-tenant authorization, or production hardening evidence is documented.','Add identity, role-based controls, rate limits, audit logging, abuse monitoring, secure headers, dependency scanning, and incident response.'),
])
para('These are design and operational risks, not a claim that a security assessment has been performed. Before processing real client matter data, carry out a privacy impact assessment, threat model, retention decision, provider-data review, and independent security test.')

heading('14. Setup and operating procedure', 1)
para('The repository documents a Python environment and dependency installation through uv. Configuration is loaded from the project-root .env into a frozen Settings object at process startup; restart the process after changing values. Provide only the provider key and optional service credentials required for the selected setup. Do not commit the .env file.')
para('Typical local sequence from the project root:')
for s in ['uv pip install -r requirements.txt', 'python -m agent.retriever.ingest', 'python scripts/validate_ingestion.py', 'streamlit run frontend/app.py']:
    p=doc.add_paragraph(style='No Spacing'); r=p.add_run(s); r.font.name='Consolas'; r.font.size=Pt(9)
para('For a retrieval-only playground, the documented command is streamlit run scripts/retrieval_playground.py. To persist normal legal-help conversations, set DATABASE_URL to PostgreSQL and use postgres_workflow(); the helper initializes the LangGraph checkpointer schema. For a fresh collection reset, the ingestion CLI provides an explicit reset operation. Resetting is destructive to that collection and should be done only after confirming its contents are reproducible from the configured PDFs.')
para('Operational checklist: verify required provider credentials; verify active provider/model and reranker; ingest and validate each collection; confirm expected sources and section/page metadata; run a smoke query per domain; check no-source abstention; check uploads are removed from temporary storage after request; and review trace settings before enabling tracing.')

heading('15. Failure modes and residual risks', 1)
table(['Failure mode','Possible effect','Existing mitigation / residual gap'], [
('Missing, outdated, or inapplicable source','Confident answer omits controlling state rule, amendment, commencement, or precedent.','Domain scoping, caveats, metadata. No automatic currentness or completeness check.'),
('Poor PDF extraction/OCR','Wrong or missing clause, section, page, or metadata.','OCR and page-aware extraction exist; sampled legal verification is still needed.'),
('Routing error or multi-domain problem','Wrong collection or dropped part of a compound dispute.','Low confidence can trigger clarification; current routing selects one domain per turn.'),
('Retrieval miss/query drift','Relevant provision absent from model context.','Hybrid, expansion, reranking, stable IDs; all remain dependent on query/corpus and ranking quality.'),
('Citation appears but does not support claim','Misleading apparent provenance.','Citation ID audit and source-restricted prompts; no semantic entailment verifier.'),
('Section correction ambiguity','A number could be changed to an unrelated unique retrieved section.','The current mechanism uses one distinct section number as replacement; semantic relevance of that replacement is not independently established.'),
('Limitation heuristic error','False reassurance or alarm.','Consumer calculation is preliminary and date-based; cause-of-action parsing and exceptions are not resolved. Other domains are always marked unclear.'),
('Safety pattern miss','Urgent disclosure enters ordinary model path.','Local regex patterns cover explicit language; indirect, multilingual, or novel wording may be missed.'),
('External provider outage/schema drift','Turn fails or structured output is unavailable.','Dependency injection enables tests; no documented production failover or service-level guarantee.'),
('Tracing or persisted state exposure','Sensitive text retained or accessed improperly.','Optional tracing and persistence exist; governance and deletion controls need implementation.'),
])

heading('16. Development roadmap and acceptance gates', 1)
table(['Priority','Work item','Completion evidence'], [
('P0','Resolve and document effective reranker defaults and add a sanitized configuration snapshot to every evaluation record.','One reproducible provider/model/config record; no secret values.'),
('P0','Run and publish a dated baseline for automated tests and all documented manual cases.','Pass/fail table, blocked cases, code revision, corpus checksum, provider status.'),
('P0','Commission legal review of source scope, commencement/effective dates, limitation language, and safety copy.','Reviewer sign-off with findings and remediation log.'),
('P1','Build expert-labeled, jurisdiction- and date-aware QA set and claim-to-source entailment grading.','Per-domain metrics, reviewer agreement, abstention results, adjudicated examples.'),
('P1','Add source freshness and state adoption metadata with monitoring and update ownership.','Source register, update process, stale-source alert, rollback record.'),
('P1','Add privacy/security controls for authentication, tenancy isolation, retention, deletion, upload handling, and tracing.','Threat model, privacy review, access-control tests, retention test, independent security assessment.'),
('P2','Benchmark latency, cost, concurrency, retrieval quality, and failure recovery; define service targets.','Load test and operational dashboard with documented SLOs.'),
('P2','Expand language and accessibility support only with reviewed translations and matching source coverage.','Domain-specific bilingual evaluation and accessibility audit.'),
])
para('Each new legal domain should pass a source-coverage gate, expert answer review, no-source/ambiguity tests, and safety/privacy review before it appears as a supported product lane.')

heading('17. Conclusion', 1)
para('LegalMind AI implements a bounded legal-information prototype with domain-specific retrieval, structured intake, source metadata, citation-ID checks, drafting, a separate debate-practice graph, and optional persistence and tracing. The repository also contains useful test specifications and deterministic anti-hallucination probes. The evidence does not yet include a completed live evaluation, expert legal validation, an authoritative and current nationwide corpus, or production security review. Its next milestone should be a reproducible, expert-adjudicated baseline and a governed corpus and privacy model, followed by targeted fixes for entailment, jurisdiction, freshness, and operational reliability.')

heading('Appendix A. Repository implementation map', 1)
table(['Concern','Principal files'], [
('Settings, source catalog, aliases','config.py'),
('Normal Legal Help graph and provider construction','agent/workflow.py; agent/schemas/state.py'),
('Intake, domain agents, drafting, compliance, safety, section correction','agent/nodes/; agent/prompts/; agent/schemas/'),
('PDF extraction, OCR, cleaning, chunking, embeddings, Chroma, retrieval, ingestion','agent/retriever/'),
('Debate Practice graph, state, schemas, prompts','agent/debate_agent/'),
('Streamlit experience, upload extraction, session and trace wiring','frontend/app.py'),
('Automated tests','tests/ and tests/integration/'),
('Evaluation specifications and template records','docs/retrieval_test_cases.md; docs/qa_test_cases.md; docs/debate_test_cases.md; docs/anti_hallucination_eval.md; evals/main_agent_runs.csv'),
('Architecture figure and setup/operation guidance','docs/architecture.svg; README.md'),
])

heading('Appendix B. Evaluation case coverage summary', 1)
table(['Case group','Coverage represented in fixtures','Key pass condition'], [
('Retrieval 1–4, 11–12','Tenancy deposit, agreement, registration, expansion, deduplication, collection boundaries.','Correct collection and relevant provenance; compare retrieval methods without requiring fixed rank.'),
('Retrieval 5–7','Consumer definition, complaint location, defective-product remedies.','Consumer-only sources with metadata and relevant passages.'),
('Retrieval 8–10','Wage definition, minimum wages, Social Security source routing.','Wages collection and source metadata, including configured folder-to-collection mapping.'),
('QA N01–N04','Deposit deductions, informal agreement, 14-month lease registration, lockout multi-turn safety.','Ask material facts, avoid categorical/local-law assumptions, preserve history, address urgency.'),
('QA N05–N09','Consumer status, repeated repair, replacement/refund, 26-month limitation, delivery dispute.','Recognize statutory issues, preserve evidence, qualify remedies and cause-of-action date.'),
('QA N10–N15','Minimum wage, deductions, delayed pay, gender pay, platform work, mixed tenancy/wage issue.','Ask location/work facts, preserve records, distinguish legal pathways, identify one-domain scope limitation.'),
('Debate case','Tenant deposit dispute from intake through confirmation, argument, response, coaching.','Argument is tied to returned excerpt labels; abstain if no relevant passage.'),
('Anti-hallucination probes','Empty evidence, unsupported section, invalid citation ID, relative timeline.','Deterministic paths behave as specified; model-driven cases require separate live evidence.'),
])

heading('Appendix C. Acronyms and terms', 1)
table(['Term','Meaning'], [
('BM25','Lexical ranked retrieval method based on term occurrence and document relevance.'),
('Chroma','Persistent vector database used for dense retrieval in this prototype.'),
('LangGraph','Graph-based orchestration and checkpointing framework used for assistant workflows.'),
('RAG','Retrieval-augmented generation: provide retrieved source passages as context to a language model.'),
('RRF','Reciprocal-rank fusion: combine ranked result lists using each result’s rank.'),
('OCR','Optical character recognition used to extract text from scanned page images.'),
('RAGAS','Evaluation toolkit proposed for answer and retrieval quality metrics; no scores are supplied here.'),
('LangSmith','Optional tracing and observability service integration.'),
])

# Body and table legibility; avoid changing the established title/heading hierarchy.
for section in doc.sections:
    section.top_margin=Inches(0.72); section.bottom_margin=Inches(0.72)
for p in doc.paragraphs:
    if p.style.name == 'Normal':
        p.paragraph_format.space_after=Pt(5)
        p.paragraph_format.line_spacing=1.08
for t in doc.tables:
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after=Pt(2)
                p.paragraph_format.line_spacing=1.0
                for run in p.runs:
                    run.font.size=Pt(8.5)

doc.core_properties.title = 'LegalMind AI Architecture, Implementation and Evaluation Report'
doc.core_properties.subject = 'Expanded academic solution report for the LegalMind AI prototype'
doc.save(PATH)
print(PATH)
