from __future__ import annotations

import io
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import faiss
import numpy as np
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

NO_ANSWER_MESSAGE = (
    "The indexed documents do not appear to contain information relevant to this question. "
    "Try rephrasing it, or lower the minimum-similarity setting if you believe the answer is in the documents."
)

# Small English stop-word list so common words do not create meaningless BM25 matches.
STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it its
    me my of on or our please she so than that the their them then there these they this those to us was we were
    what when where which who whom why will with would you your tell explain describe give about""".split()
)

_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    index: int


@dataclass
class Hit:
    chunk: Chunk
    fused_score: float
    dense_similarity: float | None  # cosine similarity; None if not among dense candidates


@dataclass
class Retrieval:
    hits: list[Hit]
    best_dense: float
    lexical_match: bool
    relevant: bool


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[\w'-]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Prefer a natural boundary near the end of the target window.
            boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + int(chunk_size * 0.6):
                end = boundary + 1
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    value = headers.get("retry-after") if headers else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _with_retry(fn: Callable[[], Any], attempts: int = 5, base_delay: float = 2.0) -> Any:
    """Retry transient API errors (rate limits, timeouts, connection drops) with backoff.

    On a rate-limit error the provider's Retry-After header is honoured (capped at 60 s), which matters on
    free tiers that limit tokens per minute.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except _RETRYABLE as exc:
            if attempt == attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            if isinstance(exc, RateLimitError):
                hinted = _retry_after_seconds(exc)
                if hinted is not None:
                    delay = min(max(hinted, delay), 60.0)
            time.sleep(delay)


class RAGEngine:
    """Hybrid retrieval-augmented generation engine: embeddings + FAISS + BM25 + rank fusion.

    Works with any OpenAI-compatible API (OpenAI, Gemini's compatibility endpoint, ...).
    `client` can be injected, which is how the offline unit tests run without an API key.
    `embedder` (a callable: list[str] -> matrix) replaces API embeddings, for example with the free local
    `LocalEmbedder`. Use it when the chat provider has no embeddings endpoint (Groq).
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        base_url: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        dense_weight: float = 0.65,
        min_similarity: float = 0.2,
        embed_batch_size: int = 32,
        client: Any = None,
        embedder: Callable[[list[str]], Any] | None = None,
    ):
        if client is None:
            if not api_key:
                raise ValueError("An API key is required.")
            client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.client = client
        self.embedder = embedder
        self.model = model
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dense_weight = dense_weight
        self.min_similarity = min_similarity
        self.embed_batch_size = embed_batch_size
        self._validate_settings()
        self.chunks: list[Chunk] = []
        self.index = None
        self.bm25 = None
        self.tokenized_chunks: list[list[str]] = []
        self.embedding_dimensions = None

    # ------------------------------------------------------------------ settings
    def _validate_settings(self) -> None:
        if not 0 <= self.dense_weight <= 1:
            raise ValueError("dense_weight must be between 0 and 1.")
        if not 0 <= self.min_similarity <= 1:
            raise ValueError("min_similarity must be between 0 and 1.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")

    # ----------------------------------------------------------------- ingestion
    def _extract_pdf(self, filename: str, pdf_bytes: bytes) -> list[tuple[int, str]]:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        extracted = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                extracted.append((page_number, text))
        if not extracted:
            raise ValueError(
                f"No extractable text was found in {filename}. If it is a scanned PDF, OCR is required."
            )
        return extracted

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        """Embed texts with the local embedder if one is set, otherwise via the API (batched, with retry)."""
        if self.embedder is not None:
            return np.array(self.embedder(texts), dtype="float32")  # copy: faiss normalises in place
        vectors = []
        for offset in range(0, len(texts), self.embed_batch_size):
            batch = texts[offset:offset + self.embed_batch_size]
            response = _with_retry(
                lambda b=batch: self.client.embeddings.create(model=self.embedding_model, input=b)
            )
            vectors.extend(item.embedding for item in response.data)
        return np.asarray(vectors, dtype="float32")

    def _embed(self, texts: list[str]) -> np.ndarray:
        matrix = self._embed_raw(texts)
        faiss.normalize_L2(matrix)
        return matrix

    @staticmethod
    def _unique_name(filename: str, seen: dict[str, int]) -> str:
        count = seen.get(filename, 0) + 1
        seen[filename] = count
        if count == 1:
            return filename
        stem, ext = os.path.splitext(filename)
        return f"{stem} ({count}){ext}"

    def index_pdfs(self, documents: list[tuple[str, bytes]]) -> None:
        pages: list[tuple[str, int, str]] = []
        seen: dict[str, int] = {}
        for filename, pdf_bytes in documents:
            name = self._unique_name(filename, seen)
            for page_number, page_text in self._extract_pdf(filename, pdf_bytes):
                pages.append((name, page_number, page_text))
        self.index_pages(pages)

    def index_pages(self, pages: list[tuple[str, int, str]]) -> None:
        """Chunk, embed and index (source, page, text) tuples. Also used directly by tests."""
        self._validate_settings()
        all_chunks: list[Chunk] = []
        for source, page, text in pages:
            for piece in _split_text(text, self.chunk_size, self.chunk_overlap):
                all_chunks.append(Chunk(piece, source, page, len(all_chunks)))
        if not all_chunks:
            raise ValueError("No text chunks were created from the uploaded PDFs.")
        matrix = self._embed([chunk.text for chunk in all_chunks])
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)
        self.embedding_dimensions = matrix.shape[1]
        self.chunks = all_chunks
        self.tokenized_chunks = [_tokenize(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    # ----------------------------------------------------------------- retrieval
    def _retrieve(self, question: str, top_k: int) -> Retrieval:
        if self.index is None or not self.chunks:
            raise RuntimeError("Index documents before asking a question.")
        n_candidates = min(len(self.chunks), max(top_k * 4, top_k))

        query_vector = self._embed([question])
        dense_scores, dense_ids = self.index.search(query_vector, n_candidates)
        dense_rank: dict[int, int] = {}
        dense_sim: dict[int, float] = {}
        for rank, (idx, score) in enumerate(zip(dense_ids[0], dense_scores[0])):
            if idx >= 0:
                dense_rank[int(idx)] = rank
                dense_sim[int(idx)] = float(score)
        best_dense = max(dense_sim.values()) if dense_sim else 0.0

        q_tokens = _tokenize(question)
        bm_scores = self.bm25.get_scores(q_tokens) if q_tokens else np.zeros(len(self.chunks))
        # Only chunks that actually share a (non-stop-word) term with the question get a lexical rank.
        bm_order = [int(i) for i in np.argsort(bm_scores)[::-1][:n_candidates] if bm_scores[i] > 0]
        bm_rank = {idx: rank for rank, idx in enumerate(bm_order)}
        lexical_match = bool(bm_rank)

        # Nothing semantically close and no shared keyword: treat the question as out of scope.
        
# Require stronger evidence when semantic similarity is low.
# A single shared word such as "list" should not be enough.

        query_terms = set(q_tokens)

        strong_lexical_match = False

        if len(query_terms) >= 2:
            for idx in bm_order:
                chunk_terms = set(self.tokenized_chunks[idx])
                shared_terms = query_terms.intersection(chunk_terms)

        # Require at least two distinct query terms
        # to appear in the same retrieved chunk.
                if len(shared_terms) >= 2:
                    strong_lexical_match = True
                    break

        if (
            best_dense < self.min_similarity
        and not strong_lexical_match
):
           return Retrieval(
        [], best_dense, lexical_match, relevant=False
    )


        # Reciprocal-rank fusion avoids mixing incomparable raw dense and BM25 scores.
        fused: dict[int, float] = {}
        for idx in set(dense_rank) | set(bm_rank):
            dense_rrf = 1.0 / (60 + dense_rank[idx]) if idx in dense_rank else 0.0
            bm_rrf = 1.0 / (60 + bm_rank[idx]) if idx in bm_rank else 0.0
            fused[idx] = self.dense_weight * dense_rrf + (1.0 - self.dense_weight) * bm_rrf
        ordered = [i for i in sorted(fused, key=lambda i: (-fused[i], i)) if fused[i] > 0][:top_k]
        hits = [Hit(self.chunks[i], float(fused[i]), dense_sim.get(i)) for i in ordered]
        return Retrieval(hits, best_dense, lexical_match, relevant=bool(hits))

    def search(self, question: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieval only (no LLM call). Used by evaluate.py for retrieval metrics."""
        self._validate_settings()
        retrieval = self._retrieve(question.strip(), top_k)
        return [
            {
                "rank": rank,
                "source": hit.chunk.source,
                "page": hit.chunk.page,
                "text": hit.chunk.text,
                "fused_score": hit.fused_score,
                "dense_similarity": hit.dense_similarity,
            }
            for rank, hit in enumerate(retrieval.hits, start=1)
        ]

    # ---------------------------------------------------------------- generation
    def _condense_question(self, question: str, history: list[dict[str, Any]]) -> str:
        """Rewrite a follow-up question as a standalone one so retrieval works.

        Falls back to the original question if the rewrite call fails.
        """
        recent = [m for m in history if m.get("role") in ("user", "assistant")][-6:]
        if not recent:
            return question
        transcript = "\n".join(f"{m['role'].upper()}: {str(m['content'])[:600]}" for m in recent)
        try:
            completion = _with_retry(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Rewrite the user's latest question so it can be understood without the conversation. "
                                "If it is already self-contained, return it unchanged. Return only the question."
                            ),
                        },
                        {"role": "user", "content": f"CONVERSATION:\n{transcript}\n\nLATEST QUESTION:\n{question}"},
                    ],
                )
            )
            text = (completion.choices[0].message.content or "").strip()
            return text or question
        except Exception:
            return question

    def ask(
        self,
        question: str,
        top_k: int = 5,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        self._validate_settings()
        standalone = self._condense_question(question, history) if history else question
        retrieval = self._retrieve(standalone, top_k)
        if not retrieval.relevant:
            return {
                "answer": NO_ANSWER_MESSAGE,
                "sources": [],
                "question": question,
                "standalone_question": standalone,
            }

        context_blocks = []
        sources = []
        for number, hit in enumerate(retrieval.hits, start=1):
            chunk = hit.chunk
            citation = f"S{number}"
            context_blocks.append(f"[{citation}] Source: {chunk.source}, page {chunk.page}\n{chunk.text}")
            sources.append({
                "citation": citation,
                "source": chunk.source,
                "page": chunk.page,
                "text": chunk.text,
                "dense_similarity": hit.dense_similarity,
            })
        context = "\n\n---\n\n".join(context_blocks)

        system_prompt = (
            "You are a careful document question-answering assistant. Answer using only the supplied context. "
            "If the context does not contain enough information, clearly say that the uploaded documents do not "
            "provide enough information. Do not invent facts. Cite claims with the source labels such as [S1]. "
            "Keep citations close to the claims they support."
        )
        user_prompt = f"QUESTION:\n{standalone}\n\nRETRIEVED CONTEXT:\n{context}\n\nAnswer the question with citations."
        completion = _with_retry(
            lambda: self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        )
        answer = completion.choices[0].message.content or "I could not generate an answer."
        return {"answer": answer, "sources": sources, "question": question, "standalone_question": standalone}
