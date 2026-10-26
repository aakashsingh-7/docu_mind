# DocuMind: AI-Powered Document Intelligence Using RAG

A local Streamlit application that lets users upload PDFs, index their text, and ask questions. The system retrieves relevant passages with **hybrid retrieval** (dense vector search with FAISS + lexical BM25, combined by weighted reciprocal-rank fusion) and asks a chat model to answer using only those passages. The UI shows source file names, page numbers, similarity scores and the retrieved text so every answer can be checked.

## Features

- Upload and index multiple PDFs (page numbers preserved; duplicate file names handled).
- Overlapping chunking with sentence-boundary preference.
- Embeddings stored in FAISS, plus a BM25 keyword index with stop-word filtering.
- Weighted reciprocal-rank fusion of the two rankings.
- **Out-of-scope detection:** a similarity cutoff returns "not in the documents" without calling the chat model.
- **Follow-up questions:** optional rewriting of "and the second one?" into a standalone question.
- Grounded answers with `[S1]`-style citations and expandable source passages.
- Works with **Groq, OpenAI, Google Gemini (OpenAI-compatible endpoint) or any OpenAI-compatible API**.
- **Free local embeddings** (no key) for providers that only offer chat, such as Groq.
- Retry with backoff on rate-limit and network errors.
- `evaluate.py` to measure retrieval on your own test questions.
- 39 offline tests (no API key needed).

## Architecture

```text
User (browser)
 |
 v
Streamlit UI (app.py)
 |
 v
RAGEngine (rag_engine.py)
 |---- PDF extraction (pypdf)
 |---- Chunking with overlap
 |---- Embeddings: provider API or free local model (fastembed)
 |---- FAISS dense index  +  BM25 lexical index
 |---- Hybrid ranking (reciprocal-rank fusion) + similarity cutoff
 |---- Follow-up rewriting + context-grounded answer generation
 |
 v
Answer + citations + source snippets
```

Diagrams for a report (architecture, DFD level 0/1, ER) are in `docs/diagrams.md`.

## Setup

Python 3.10–3.12 recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-local.txt   # needed for free local embeddings (Groq setup below)
cp .env.example .env                # Windows PowerShell: Copy-Item .env.example .env
```

Edit `.env`: set `LLM_PROVIDER` and `LLM_API_KEY`.

### Free setup with Groq (recommended if you have no paid key)

1. Create a free account and API key in the Groq console (console.groq.com, "API Keys").
2. In `.env` set `LLM_PROVIDER=groq` and `LLM_API_KEY=<your key>`.
3. Run `streamlit run app.py`. Chat goes to Groq; embeddings are computed on your own computer.

Why two parts: Groq's API serves chat models but not embeddings, so DocuMind computes embeddings locally with `sentence-transformers/all-MiniLM-L6-v2`. The first index build downloads that model (roughly tens of MB) and needs internet once; after that embeddings run offline. Only the question and the retrieved passages go to Groq, not the full PDFs.

**Best model:** the default is `openai/gpt-oss-120b`, the largest chat model Groq lists as a production model. In the sidebar, click **Load models from provider**: the app asks Groq which models your key can use, hides speech, safety and agentic models, and selects the best available one (order: gpt-oss-120b, llama-3.3-70b-versatile, qwen3-32b, gpt-oss-20b, llama-3.1-8b-instant). You can still choose another model from the list.

Free-tier notes: Groq limits requests and tokens per minute and per day, and the limits change; a third-party summary put gpt-oss-120b at roughly 8,000 tokens per minute on the free plan, which I could not verify. DocuMind waits and retries when Groq says "rate limited" (it honours the Retry-After header). If you still hit limits, lower "Retrieved passages", or pick `llama-3.1-8b-instant`, which is lighter. Model names can be renamed or retired; if Groq answers "model not found", click **Load models from provider** again.

**Keep your key private:** put it only in `.env` (already in `.gitignore`), never in code, screenshots, the report or a GitHub repo. If a key was ever pasted somewhere public or into a chat, delete it in the Groq console and create a new one.

### Choosing a provider

| Provider | `LLM_PROVIDER` | Notes |
|---|---|---|
| Groq | `groq` | Free tier. Chat only; embeddings run locally (see above). |
| OpenAI | `openai` | Paid API key. Defaults: `gpt-4o-mini`, `text-embedding-3-small`. |
| Google Gemini | `gemini` | Uses Gemini's OpenAI-compatible endpoint, which supports chat and embeddings. Create a key in Google AI Studio; check Google's current free-tier limits. |
| Anything else | `custom` | Set `LLM_BASE_URL`, `CHAT_MODEL`, `EMBEDDING_MODEL`. The provider must offer both chat and embeddings endpoints. |

Model names change. The defaults in `config.py` were not tested against live keys or a real model download when this project was prepared; the code paths are covered by offline tests with fake clients. If a provider rejects a model name, look up its current model list and change it in the sidebar or `.env`.

## Run the app

```bash
streamlit run app.py
```

Upload PDFs, click **Build / rebuild index**, then ask questions. Sidebar settings marked "applied live" take effect on the next question; chunk size, overlap and embedding model need a rebuild.

## Run tests

```bash
pytest -q
```

The tests use fake clients and a fake local embedder, so they run offline and cost nothing. They cover validation, chunking, hybrid retrieval, the BM25 regression, the out-of-scope cutoff, follow-up rewriting, real PDF extraction and an app smoke test.

## Measure it on your own documents

1. Put your PDFs in a folder, for example `my_pdfs/`.
2. Copy `eval/questions_template.csv` to `eval/questions.csv`, delete the EXAMPLE rows and write 20–30 questions. For each, record the PDF name and the page(s) where the answer really is. Include a few questions whose answers are **not** in your documents (`answerable = no`).
3. Run:

```bash
python evaluate.py --pdfs my_pdfs --questions eval/questions.csv
python evaluate.py --pdfs my_pdfs --questions eval/questions.csv --with-answers   # also generates answers
```

It prints counts (for example "correct page in top-5: X of N") and writes `eval/results.csv`. With `--with-answers`, fill in the two manual columns (answer correct? faithful to sources?) yourself. Try different `--chunk-size`, `--top-k` and `--dense-weight` values and report what you actually observe, including failures.

## How it works

1. **Ingestion:** each PDF is read page by page; pages with extractable text keep their page numbers.
2. **Chunking:** page text is split into overlapping chunks.
3. **Embedding and indexing:** chunks are embedded, normalised and stored in a FAISS inner-product index (cosine similarity after normalisation). A BM25 index is built over stop-word-filtered tokens.
4. **Retrieval:** the question is embedded and searched in FAISS; BM25 finds keyword matches (only real matches count). If the best semantic similarity is below the cutoff and there is no keyword match, the question is treated as out of scope.
5. **Fusion:** the two ranked lists are combined by weighted reciprocal-rank fusion.
6. **Generation:** retrieved chunks go into a prompt that instructs the model to answer only from the context and to cite `[S#]` labels.
7. **Display:** answer, labelled sources, page numbers, similarity and snippets.

## Project structure

```text
documind/
├── app.py              Streamlit UI
├── rag_engine.py       ingestion, hybrid retrieval, generation
├── config.py           provider presets and .env settings
├── embedders.py        free local embeddings (fastembed)
├── utils.py            PDF upload validation
├── evaluate.py         retrieval evaluation on your own questions
├── requirements.txt
├── requirements-local.txt   optional: local embeddings
├── .env.example
├── .gitignore
├── pytest.ini / conftest.py
├── README.md
├── CHANGES.md          what was fixed in this version
├── REPORT_GUIDE.md     content guide for a written report
├── docs/diagrams.md    Mermaid diagrams (architecture, DFD, ER)
├── eval/questions_template.csv
└── tests/
```

## Important limitations

- Text sent to the chat provider includes your questions and retrieved passages; with API embeddings, all extracted text is sent too. Review data rules before uploading sensitive documents.
- Answers can still be wrong. Check citations against the displayed passages; the model may not attach a citation to every claim.
- The similarity cutoff is a heuristic. Tune it with your own questions; the right value depends on the embedding model.
- The minimum-similarity check works on whole questions. Very short or vague questions may be rejected or may match loosely.
- OCR for scanned PDFs, persistent storage, authentication, access control and deployment hardening are not implemented. The index lives in Streamlit session memory and must be rebuilt after a restart.
- API use may incur charges. Check current pricing and limits.

## Academic integrity note

Use this project as a working prototype and make sure you understand the code before presenting it. Describe it accurately as the project you built. Do not claim that an internship took place if it did not, fabricate an internship certificate, or report test results you did not actually obtain.
# docu_mind
