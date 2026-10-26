# Report-writing guide for DocuMind RAG project

Use this as a content guide alongside your institute's required report format. Replace bracketed items with accurate information and adapt chapter numbering to the institute template.

## Suggested title
**AI-Powered Document Intelligence System Using Retrieval-Augmented Generation (RAG)**

## Abstract (write after implementation and testing)
Describe the problem of finding answers inside lengthy PDFs, the objective of building a document question-answering prototype, PDF extraction and chunking, embedding generation, FAISS and BM25 hybrid retrieval, context-grounded response generation, and source/page display. State the actual tests you performed and the actual outcomes. Do not invent accuracy percentages.

## Problem statement
Users may need to manually search large PDF documents to find relevant information. Keyword search can miss semantically related passages, while a general-purpose language model may answer without grounding in the supplied documents. The prototype combines semantic and lexical retrieval with context-grounded answer generation and source display.

## Objectives
- Build a PDF ingestion pipeline.
- Extract text while retaining page numbers.
- Divide text into overlapping chunks.
- Create and index text embeddings.
- Retrieve relevant passages using FAISS and BM25.
- Generate answers from retrieved context.
- Display source passages to support verification.
- Test the prototype and document limitations.

## Functional requirements
1. The user can upload one or more PDF files.
2. The system validates file extension and PDF header.
3. The system extracts page text and creates chunks.
4. The system builds dense and lexical search indexes.
5. The user can submit a natural-language question.
6. The system retrieves relevant chunks and generates a grounded answer.
7. The system shows source names, page numbers, and snippets.
8. The user can clear the current index.

## Non-functional requirements
- Usability: browser-based Streamlit interface.
- Maintainability: UI, retrieval engine, and validation are separated into modules.
- Reliability: invalid files and missing keys produce visible errors.
- Traceability: source names and page numbers are displayed.
- Privacy: users are warned that extracted text is sent to the configured API.
- Performance: indexing and retrieval depend on document size and API latency.

## System design
Client: Streamlit interface in the browser.
Application layer: `app.py` and `RAGEngine`.
Ingestion layer: `pypdf` extraction and chunking.
Retrieval layer: OpenAI embeddings, FAISS, BM25, and weighted rank fusion.
Generation layer: OpenAI chat completion using retrieved context.
Output: answer, source labels, page numbers, and source snippets.

## Data-flow description
1. User uploads PDFs.
2. The application validates files.
3. PDF pages are extracted and split into chunks.
4. Chunk embeddings are generated and indexed in FAISS; chunk tokens are indexed by BM25.
5. User submits a question.
6. The question is embedded and searched in both indexes.
7. The ranked passages are combined and supplied to the language model.
8. The model returns a context-grounded answer.
9. The interface displays the answer and source passages.

## Test plan
Create and execute a table of tests with columns: Test ID, Input/Action, Expected Result, Actual Result, Status.
Include valid PDF upload, invalid file rejection, multiple PDFs, question with an answer in the document, question absent from the document, page/source display, empty question, and API error handling. Fill Actual Result and Status only after running each test.

## Results and conclusion
Report what worked in your own run, which PDFs and questions you tested, and any failures. A reasonable conclusion is that the prototype demonstrates the core RAG pipeline, while OCR, persistence, authentication, systematic evaluation, and production deployment remain future work. Adjust this conclusion to match observed results.

---

# Additions for this version

## Extra functional requirements
9. The system answers "not in the documents" for out-of-scope questions without calling the language model.
10. The system can rewrite follow-up questions into standalone questions.
11. The user can choose an OpenAI-compatible provider (for example Groq) and model names, and can compute embeddings locally instead of through an API.
12. Transient API errors (rate limits, timeouts) are retried with backoff.

## Extra test-plan rows (fill Actual Result and Status yourself after running)
- Question with its answer in the document, expect correct page in the sources.
- Question not in the document, expect the "not in the documents" message and no sources.
- Follow-up question after a first answer, expect a rewritten "Searched for" line.
- Duplicate PDF names, expect numbered suffix.
- Slider changes (top-k, dense weight, minimum similarity) apply to the next question without rebuilding.

## Results chapter
Run `python evaluate.py` (see README) and report the printed counts together with the settings line it prints: number of documents, chunks, questions, chunk size, overlap, top-k, dense weight and minimum similarity. Compare at least two settings (for example chunk size 500 vs 1000) and discuss the failures you see. Do not report numbers you did not measure.

## Diagrams
Architecture, DFD level 0 and 1, and an ER diagram (logical data model) are in `docs/diagrams.md`.

## Mapping to a typical college report order
Introduction → Requirement Analysis (functional and non-functional requirements above) → System Design (architecture, DFD, ER from `docs/diagrams.md`) → Algorithm (chunking, embeddings, cosine similarity, BM25, reciprocal-rank fusion, prompt) → Screenshots (appendix) → Results and observations (from `evaluate.py` and your test table) → Conclusions and future scope (OCR, persistence, authentication, re-ranking, larger evaluation). Title page, certificate and declaration must reflect what is actually true for you.
