# What changed in this version

| # | Problem in the first version | Fix |
|---|---|---|
| 1 | BM25 gave a rank to chunks with a score of 0, adding noise to the fused ranking | Only chunks that share a keyword with the question get a lexical rank; zero-score fusion results are dropped. Regression test included. |
| 2 | Dense-weight and chat-model sliders did nothing after indexing | They are now applied on every question. Only chunk size, overlap and embedding model need a rebuild (labelled in the sidebar). |
| 3 | Earlier chat messages showed sources without their `[S1]` labels | Every message now shows labelled sources, plus the similarity score when available. |
| 4 | Out-of-scope questions still sent irrelevant chunks to the LLM | A minimum-similarity cutoff (plus keyword check) returns "not in the documents" without calling the chat model. |
| 5 | Follow-up questions ("and the second one?") failed | Optional question rewriting using the last few messages. |
| 6 | Required a paid OpenAI key | Works with any OpenAI-compatible API. Built-in presets: Groq (free tier), Gemini, OpenAI. |
| 6b | Groq has no embeddings endpoint | Added free local embeddings (`embedders.py`, fastembed, all-MiniLM-L6-v2); Groq preset uses them by default. |
| 7 | No retry on rate limits | Backoff retry on rate-limit, timeout and connection errors; honours the provider's Retry-After header (free tiers). |
| 7b | Hard-coded model names go stale | "Load models from provider" button lists the models your key can use and auto-selects the best (Groq default: openai/gpt-oss-120b). |
| 8 | Stop-words created meaningless keyword matches | Small stop-word filter in the tokenizer. |
| 9 | Duplicate file names mixed up sources | Duplicates get a numbered suffix. |
| 10 | README listed files that were missing | Added `.env.example`, `.gitignore`, `tests/`, `pytest.ini`. |
| 11 | No way to produce real results for the report | `evaluate.py` measures retrieval on your own question set and writes a CSV. |

Tests: 39 offline tests (no API key needed) cover chunking, validation, retrieval, the cutoff, follow-up rewriting, local embeddings, the Groq preset, real PDF extraction and app smoke tests.
