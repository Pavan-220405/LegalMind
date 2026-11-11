from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT = r"F:\AI_Lawyer\output\docx\legalmind-solution-report-two-page-expanded.docx"
d = Document()
sec=d.sections[0]
sec.page_width=Inches(8.27); sec.page_height=Inches(11.69) # A4
sec.top_margin=Inches(.48); sec.bottom_margin=Inches(.45)
sec.left_margin=Inches(.58); sec.right_margin=Inches(.58)
normal=d.styles['Normal']; normal.font.name='Cambria'; normal.font.size=Pt(9.0)
normal.paragraph_format.space_after=Pt(3.0); normal.paragraph_format.line_spacing=1.03
for sty,size,color in [('Title',21,'18324F'),('Heading 1',11.5,'18324F'),('Heading 2',9.5,'315A78')]:
    s=d.styles[sty]; s.font.name='Cambria'; s.font.size=Pt(size); s.font.bold=True
    s.font.color.rgb=RGBColor.from_string(color)
    s.paragraph_format.space_before=Pt(4 if sty!='Title' else 0)
    s.paragraph_format.space_after=Pt(2)

def shade(cell, fill):
    tcPr=cell._tc.get_or_add_tcPr(); sh=OxmlElement('w:shd'); sh.set(qn('w:fill'),fill); tcPr.append(sh)
def repeat(row):
    trpr=row._tr.get_or_add_trPr(); tag=OxmlElement('w:tblHeader'); tag.set(qn('w:val'),'true'); trpr.append(tag)
def table(headers, rows, widths=None):
    t=d.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'
    t.autofit=False
    if widths:
        for col,w in zip(t.columns,widths): col.width=Inches(w)
    repeat(t.rows[0])
    for i,h in enumerate(headers):
        c=t.rows[0].cells[i]; c.text=h; shade(c,'18324F')
        for r in c.paragraphs[0].runs: r.bold=True; r.font.color.rgb=RGBColor(255,255,255); r.font.size=Pt(7.4)
    for vals in rows:
        cells=t.add_row().cells
        if widths:
            for c,w in zip(cells,widths): c.width=Inches(w)
        for i,v in enumerate(vals):
            cells[i].text=v; cells[i].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cells[i].paragraphs:
                p.paragraph_format.space_after=Pt(1.4); p.paragraph_format.line_spacing=1.0
                for r in p.runs: r.font.name='Cambria'; r.font.size=Pt(7.8)
    d.add_paragraph().paragraph_format.space_after=Pt(0)
    return t
def para(text, boldlead=None):
    p=d.add_paragraph()
    if boldlead and text.startswith(boldlead):
        p.add_run(boldlead).bold=True; p.add_run(text[len(boldlead):])
    else: p.add_run(text)
    return p
def bullet(text):
    p=d.add_paragraph(style='List Bullet'); p.paragraph_format.left_indent=Inches(.16); p.paragraph_format.first_line_indent=Inches(-.1); p.paragraph_format.space_after=Pt(1); p.add_run(text); return p

p=d.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run('LegalMind AI')
p=d.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(5)
r=p.add_run('Two-page solution report | Architecture, implementation and evaluation'); r.bold=True; r.font.size=Pt(9); r.font.color.rgb=RGBColor.from_string('526579')

d.add_heading('Purpose and scope',1)
para('LegalMind AI is a prototype for Indian legal information and debate practice. It focuses on rent and tenancy, consumer protection, and wages/social security: it turns a user narrative into a structured issue, retrieves passages from a bounded legal corpus, explains practical options, and can draft a communication or support argument practice. It is an information tool, not legal representation or independently validated legal advice. State law, commencement, current amendments, forum, contract terms, dates, and facts can change the result; consequential actions require local legal review.')

d.add_heading('System design',1)
table(['Component','Implemented behavior'],[
('Legal Help workflow','LangGraph records a turn; screens explicit self-harm/imminent-threat patterns locally; uses structured model intake for issue, domain, confidence, explicit dates, evidence and urgency; clarifies uncertain routing; sends supported domains to a retrieval agent; optionally drafts a notice; checks citation IDs, limitation uncertainty and urgency; corrects unsupported section references; then composes the response.'),
('Domain retrieval','Domain-limited Chroma collections prevent cross-domain retrieval. Optional LLM query expansion adds three phrasings while preserving material factual tokens. Dense BGE-M3 and BM25 candidate lists (20 each by default) are combined with 0.5/0.5 reciprocal-rank fusion; optional Cohere or Voyage reranking returns up to six passages.'),
('Debate Practice','Separate LangGraph flow confirms the practice issue, user position and supported domain; retrieves excerpts; argues the opposing side with source labels; maintains the exchange; and coaches on request using retrieved context. Empty retrieval triggers abstention.'),
('Interface and state','Streamlit exposes Legal Help and Debate Practice, a new-conversation control, streamed pipeline activity, citations and PDF attachment up to 10 MB. PDF text is extracted in a temporary directory. UI state defaults to in-memory checkpoints and is lost at process stop; PostgreSQL checkpointing is available for the normal workflow.'),
])
para('Runtime configuration is environment-backed. The checked-in Settings defaults select Voyage rerank-3-lite, while README examples describe Cohere rerank-v5.0-fast; environment values may override either. A reproducible run must record its effective provider, model and retrieval settings, without recording credentials.')

d.add_heading('Request and retrieval detail',1)
para('At intake, the supervisor uses the current message, up to eight prior turns and extracted attachment text to populate structured state. It records domain and extraction confidence, the problem statement, raw and explicitly parsed timeline, user-stated evidence separately from PDF-extracted evidence, and urgency. A relative date such as “about three months ago” stays raw unless an explicit date is identified. Low-confidence or uncertain routing returns a focused clarification instead of selecting a legal collection.')
para('Retrieval is domain-scoped: rent/property maps to rent_law, consumer protection to consumer_law, and wages to wages_law; supported aliases normalize to these configured names. BM25 is reconstructed lazily from persisted chunks. Expanded results are deduplicated by stable chunk ID, and reranking uses the original question. Each displayed source can retain title, section, page range, path and chunk ID. The reranker is a relevance ranker, not a legal authority or truth check.')
table(['Setting','Default / behavior'],[
('Embedding','BAAI/bge-m3, CPU by default; batch size 8.'),
('Candidate retrieval','20 dense + 20 BM25 candidates; equal 0.5/0.5 fusion weights.'),
('Query expansion','Enabled by default; three alternatives, preserving years, dates, amounts, durations and section references.'),
('Reranking','Enabled; final top six. Effective provider is environment-configurable; README and Settings defaults differ.'),
('Models and storage','Groq or Google chat provider; local persistent Chroma; PostgreSQL checkpointer can persist normal workflow threads.'),
])

d.add_heading('Corpus and processing',1)
table(['Collection','Configured source set'],[
('Consumer','Consumer Protection Act, 2019; General Rules, 2020; Consumer Commission Regulations, 2020.'),
('Wages/social security','Code on Wages, 2019; Code on Wages (Amendment), 2026; Code on Social Security, 2020.'),
('Rent/property','Model Tenancy Act, 2021; Registration Act, 1908; Transfer of Property Act, 1882.'),
])
para('PDF extraction is page-aware and can invoke OCRmyPDF/Tesseract (configured English and Hindi). Conservative cleaning and structure-aware chunking retain section/page boundaries; long provisions split at subsections or paragraphs, with roughly 650 estimated tokens per chunk and overlap for oversized text. Chunks carry source hash, path, document/domain, page range, section/chapter, chunk ID and embedding metadata. Deterministic IDs let re-ingestion skip unchanged documents and replace changed chunks; BM25 is rebuilt from persistent chunks and invalidated after ingestion. The source set is not exhaustive or automatically refreshed; inclusion does not prove currentness, commencement, state adoption, or applicability.')
para('Corpus routing is explicitly configured in config.py. The Code on Social Security PDF is physically stored under docs/consumer but assigned to the wages collection. No case-law database, live official-source synchronization, automatic amendment detector, or state-by-state adoption map is included. Corpus governance should track issuing authority, official publication URL, download date, checksum, amendment and commencement status, jurisdiction, and replacement history. Sample extracted pages and section labels should be checked against source PDFs after every update.')

d.add_page_break()
p=d.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run('Evaluation, safeguards and next steps')

d.add_heading('Safeguards and their limits',1)
para('Domain prompts restrict statutory propositions to retrieved excerpts and require uncertainty; intake separates user facts from uploaded-PDF facts and preserves relative dates; drafting uses placeholders for missing facts. Agents remove citation IDs not in the retrieved list. Compliance flags invalid or absent citations, adds limitation warnings and urgency flags. A narrow section checker replaces unsupported numbers only when retrieved metadata yields one section, otherwise removing the number. The debate path abstains without useful excerpts. These checks validate citation-ID membership, not whether cited text entails a claim; there is no general semantic claim verifier, current-law checker, or jurisdiction validator. The safety pattern screen is not clinical assessment and can miss indirect or multilingual disclosures.')

d.add_heading('Verification evidence',1)
table(['Evidence','What is currently available'],[
('Automated coverage','Tests exist for retrieval, Chroma persistence, ingestion metadata, workflows and debate. A fresh passing run is not claimed in this report.'),
('Manual fixtures','12 retrieval cases compare retrieval modes, provenance and collection boundaries; 15 legal QA cases span tenancy, consumer, wage/social-security and mixed-scope problems; a debate case checks intake through coaching.'),
('Anti-hallucination probes','Deterministic no-source, invalid-citation and section-correction outputs are documented. Provider-backed live transcripts were not captured because the endpoint was unreachable.'),
('Measured results','The evaluation CSV is an empty template. No accuracy, entailment, RAGAS score, latency, cost, benchmark sample, or independent legal review is recorded.'),
])
para('The evidence supports claims about implemented code paths and specified checks, not measured legal quality, accuracy, or production readiness. Before claiming performance, freeze code/corpus/configuration, run the cases, save prompts, retrieved chunks and answers, and have qualified reviewers score claim support, citation entailment, jurisdiction/date handling, omissions, calibration, safety and usefulness. Report per-domain results, sample size, reviewer agreement and failures; separate retrieval, answer, clarification, abstention and safety measures.')
table(['Fixture group','Scenarios represented'],[
('Tenancy','Deposit deductions and evidence; unsigned/message-based agreement; 14-month lease registration; multi-turn lockout and safety; deposit-retention debate.'),
('Consumer','Commercial-purpose consumer status; repeated failed repair; replacement/refund; explicit 26-month delay and cause-of-action uncertainty; non-delivery and platform/seller roles.'),
('Wages/social security','Minimum wage and location/category; tool-damage deduction; delayed pay records; gender pay comparator; gig/platform classification.'),
('Cross-cutting','Compound tenancy-plus-wage issue, low-confidence clarification, exact section retrieval, collection isolation, no-source abstention and provenance checks.'),
])
para('The retrieval suite also compares dense, BM25 and hybrid results on repeat runs, checks stable chunk IDs and source metadata, toggles multi-query/reranking, and verifies that expansion results stay inside the chosen collection. These checks define expected behavior but do not assert fixed result ranks. RAGAS is proposed for faithfulness, answer relevance, context precision/recall and factual correctness where the installed version supports it; expected reference contexts must be actual excerpts from the PDFs, not only prose descriptions.')

d.add_heading('Risks, privacy and operational readiness',1)
para('Material failure modes include stale or inapplicable sources, OCR/section errors, routing or retrieval misses, reranker misordering, unsupported claims with valid citation IDs, incomplete limitation logic, provider/schema outages and safety-pattern misses. Consumer limitation screening is explicitly preliminary and date-based; other domains remain unclear. Multi-issue prompts may be routed to one domain at a time. External model calls process user prompts and retrieved/uploaded text; optional LangSmith tracing may retain legal narratives and excerpts. In-memory checkpoints clear at process stop, while PostgreSQL and local Chroma persist data. Authentication, tenant isolation, retention/deletion controls and an independent security review are not established in the repository.')
table(['Operational area','Current state and required control'],[
('Secrets and providers','Keys come from environment settings. Keep .env out of version control, use scoped secret storage and rotate keys; record provider/model but never credentials.'),
('Uploads and prompts','PDFs up to 10 MB are extracted from a temporary file; extracted text can enter model prompts/checkpoints. Define deletion, malformed-file limits, user notice and provider data handling.'),
('Persistence and traces','UI checkpoints default to memory; optional PostgreSQL and local Chroma persist. LangSmith tracing is optional and may capture inputs/outputs. Set access controls, retention, encryption and deletion before real client data.'),
('Release operations','Ingest PDFs, run ingestion validation, smoke-test one query per collection and empty-retrieval abstention, then record corpus hashes and effective settings. Add auth, isolation, monitoring, incident response and dependency review before deployment.'),
])

d.add_heading('Prioritized next steps',1)
bullet('Resolve reranker defaults and capture sanitized effective configuration in run records.')
bullet('Produce a dated automated and manual baseline with corpus checksums, model/provider, settings, failures, latency and cost.')
bullet('Have legal experts review source scope, commencement, state applicability, limitation warnings and safety language; add versioned official sources and update ownership.')
bullet('Add expert-labelled, jurisdiction/date-aware cases and claim-to-source entailment grading; test abstention, compound issues and adversarial inputs.')
bullet('Complete privacy/security design: authentication, tenant isolation, upload limits/handling, encryption, retention/deletion, trace minimization and incident response.')
bullet('Benchmark reliability, concurrency, latency and cost before deployment; gate each new legal domain on source coverage and expert review.')

d.add_heading('Conclusion',1)
para('LegalMind AI has a clear prototype architecture: domain-scoped hybrid retrieval, structured intake, citations and metadata, drafting and compliance steps, a separate debate coach, and optional persistence/tracing. Its next milestone is a reproducible, expert-adjudicated evaluation supported by governed, current sources and privacy/security controls. Until then, users should verify legal propositions and deadlines with a qualified local professional before acting.')

# Keep all dense blocks legible, with table rows kept together where possible.
for t in d.tables:
    for row in t.rows:
        trPr=row._tr.get_or_add_trPr(); cant=OxmlElement('w:cantSplit'); trPr.append(cant)
d.core_properties.title='LegalMind AI Two-Page Solution Report'
d.core_properties.subject='Concise architecture, implementation, evaluation and readiness report'
d.save(OUT)
print(OUT)
