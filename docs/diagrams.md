# Diagrams for the report

Paste each block into https://mermaid.live to render and export as PNG/SVG, or use any Mermaid-aware editor.
Note: DocuMind keeps its index in memory (no database), so the ER diagram below is a *logical data model* of what the
program holds while running. Say so in your report.

## 1. Client-server architecture

```mermaid
flowchart LR
    U[User in browser] -->|upload PDFs, ask questions| UI[Streamlit client - app.py]
    UI --> E[RAGEngine - rag_engine.py]
    E --> P[PDF extraction - pypdf]
    E --> C[Chunking with overlap]
    E --> V[(FAISS dense index)]
    E --> B[(BM25 keyword index)]
    E -->|chat requests| API[OpenAI-compatible chat API e.g. Groq]
    API --> E
    E -->|embeddings| L[Local embedding model or provider API]
    L --> E
    E -->|answer, citations, passages| UI
    UI --> U
```

## 2. DFD level 0

```mermaid
flowchart LR
    User((User)) -->|PDF files| S[DocuMind RAG system]
    User -->|Questions| S
    S -->|Answers with citations and source passages| User
    S <-->|Text to embed / prompts and generated text| API((LLM API provider))
```

## 3. DFD level 1

```mermaid
flowchart TD
    User((User)) -->|PDF files| P1[1. Validate and extract text]
    P1 -->|page text| P2[2. Chunk text]
    P2 -->|chunks| P3[3. Embed and index]
    P3 --> D1[(Vector index + BM25 index)]
    User -->|Question| P4[4. Rewrite follow-up if needed]
    P4 -->|standalone question| P5[5. Hybrid retrieval]
    D1 --> P5
    P5 -->|top passages| P6[6. Generate grounded answer]
    P6 -->|answer + sources| User
    P3 <-->|embeddings| API((LLM API))
    P5 <-->|query embedding| API
    P6 <-->|prompt / answer| API
```

## 4. ER diagram (logical data model)

```mermaid
erDiagram
    DOCUMENT ||--o{ PAGE : contains
    PAGE ||--o{ CHUNK : "split into"
    CHUNK ||--|| EMBEDDING : has
    SESSION ||--o{ MESSAGE : has
    MESSAGE ||--o{ RETRIEVED_SOURCE : cites
    CHUNK ||--o{ RETRIEVED_SOURCE : "returned as"

    DOCUMENT { string filename }
    PAGE { int page_number }
    CHUNK { int chunk_id string text }
    EMBEDDING { float_vector vector }
    SESSION { string session_id }
    MESSAGE { string role string content }
    RETRIEVED_SOURCE { string citation_label float similarity }
```
