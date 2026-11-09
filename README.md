# Legal PDF ingestion and retrieval

This repository provides the document-ingestion and retrieval foundation for
the legal RAG system. Domain agents, routing, legal answer generation, and
drafting are intentionally outside this layer.

## Install

```powershell
uv pip install -r requirements.txt
```

The vector database is persistent local Chroma. The first embedding and query
expansion calls may load models. Cohere reranking requires `COHERE_API_KEY`;
Voyage reranking requires `VOYAGE_API_KEY`.

## Configuration

Application settings are loaded once from the project-root `.env` into the
shared `settings` object in [config.py](config.py). Workflow, retrieval,
embeddings, OCR, and the Streamlit app read configuration from this object.
Restart the process after changing `.env`. The older
`agent.retriever.config` module re-exports aliases for compatibility.

```text
VECTOR_DB=chroma
CHROMA_PERSIST_DIRECTORY=./data/chroma
DENSE_CANDIDATES=20
BM25_CANDIDATES=20
RETRIEVAL_CANDIDATES=20
DENSE_WEIGHT=0.5
BM25_WEIGHT=0.5
MULTI_QUERY_ENABLED=true
MULTI_QUERY_COUNT=3
RERANKER_ENABLED=true
RERANKER_PROVIDER=voyage
RERANKER_MODEL=rerank-v5.0-fast
VOYAGE_RERANKER_MODEL=rerank-3-lite
RERANK_TOP_K=6
HF_EMBEDDING_MODEL=BAAI/bge-m3
LEGAL_RAG_LLM_PROVIDER=groq
LEGAL_MODEL=openai/gpt-oss-20b
SMALL_MODEL=openai/gpt-oss-20b
```

`CONSUMER_COLLECTION_NAME`, `LABOUR_COLLECTION_NAME`, and
`PROPERTY_COLLECTION_NAME` may override `consumer_law`, `wages_law`, and
`rent_law`. Public labels `consumer`, `labour`, `property`, `wages`, and `rent`
are accepted too.

## Collection architecture and ingestion

Chroma persists all domain collections under `CHROMA_PERSIST_DIRECTORY`.
The existing embedding model, PDF extraction/OCR, conservative cleaning,
domain classification, structure-aware chunking, and metadata are retained.
Each chunk receives a deterministic `chunk_id`; repeated ingestion skips
unchanged sources and replaces only changed chunks.

```powershell
python -m agent.retriever.ingest
python -m agent.retriever.ingest --collection rent
python -m agent.retriever.ingest --collection consumer
python scripts/validate_ingestion.py
streamlit run scripts/retrieval_playground.py
```

## Retrieval

The domain-scoped API returns LangChain `Document` objects with source and
statutory metadata:

```python
from agent.retriever.retrieval import retrieve_documents

results = retrieve_documents(
    query="My landlord is refusing to return my security deposit",
    collection_name="rent_law",
)
```

Dense Chroma similarity search and a BM25 index reconstructed from persisted
chunks are combined by LangChain `EnsembleRetriever` using weighted reciprocal
rank fusion. Query expansion searches the original plus three LLM-generated
alternatives and merges by stable chunk ID. The configured LangChain reranker
reranks against the original query and returns six documents by default. Set
Voyage Lite (`VOYAGE_RERANKER_MODEL`) is the primary reranker by default. If
Voyage returns a rate-limit response, the pipeline retries that ranking request
with Cohere Fast (`RERANKER_MODEL`). Other Voyage errors are surfaced. Set
`RERANKER_PROVIDER=cohere` to use Cohere alone. BM25 is
rebuilt on demand after process restart and invalidated when ingestion changes
a collection.

For tests or diagnostics, disable model-backed stages:

```python
results = retrieve_documents(
    "security deposit refund", "rent_law",
    multi_query_enabled=False, reranker_enabled=False,
)
```

Lower-level components are `HybridRetriever`, `LegalMultiQueryRetriever`,
`CohereReranker`, and `VoyageReranker` in `agent.retriever.retrieval`.

## Legal assistance workflow

`agent.workflow.build_legal_workflow` assembles the structured supervisor,
domain routing, rent/consumer/wages retrieval agents, general fallback, drafting,
phase-one compliance, and final conversation node. Model and retrieval
dependencies are injectable. The supervisor routes uncertain or low-confidence
domains to a focused clarification turn. Retrieval is restricted to the selected
domain collection.

For local use, configure `LEGAL_RAG_LLM_PROVIDER` as `groq` or `google`, set the
corresponding provider key, and set `DATABASE_URL` to a PostgreSQL database.
`agent.workflow.postgres_workflow()` opens the LangGraph PostgreSQL checkpointer
and runs its setup migration. Invoke turns with a stable per-conversation
`thread_id`:

```python
from agent.workflow import invoke_turn, postgres_workflow

with postgres_workflow() as graph:
    result = invoke_turn(
        graph,
        thread_id="conversation-123",
        user_message="My landlord has not returned my security deposit.",
    )
    print(result["final_response"])
```

### Streamlit pipeline playground

Run the interactive pipeline check from the repository root:

```powershell
streamlit run frontend/app.py
```

The Streamlit browser session creates and retains its own thread ID. **New
conversation** generates a different ID. The app opens in temporary in-memory
mode so you can try the pipeline immediately; choose PostgreSQL for durable
checkpoints once the configured database exists and is reachable. In-memory
checkpoints are cleared when the app process stops. The sidebar shows Chroma
collection counts; ingest the legal PDFs first if the collections are empty.

Conversation messages use LangGraph's `messages` reducer and are persisted in
the checkpoint; there is no separate manually maintained history field. The
workflow's limitation check is a cautionary heuristic: it only compares an
explicitly parsed date for consumer complaints and always reports the Consumer
Protection Act section 69 exception and verification warning. Rent, wages, and
other limitation periods remain unclear pending local legal review. Compliance
currently audits citations, limitation warnings, urgency, and the general
fallback disclaimer; unsupported claims and addressee/jurisdiction checks are
future work.

## Testing

```powershell
python -m unittest discover -s tests -v
```

Ordinary retrieval tests use synthetic documents and injected fake LLMs and
rerankers; the Chroma persistence check uses an isolated temporary directory.
The workflow integration test uses fake models and an in-memory checkpointer,
so it does not require provider API keys or a running PostgreSQL instance.
Manual query cases are listed in `docs/retrieval_test_cases.md`.
