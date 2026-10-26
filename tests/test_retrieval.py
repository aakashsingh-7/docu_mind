import pytest

from rag_engine import NO_ANSWER_MESSAGE, RAGEngine
from tests.helpers import PAGES, FakeClient, make_pdf


def build(client=None, **kwargs):
    client = client or FakeClient()
    engine = RAGEngine(client=client, **kwargs)
    engine.index_pages(PAGES)
    return engine, client


def test_relevant_chunk_is_ranked_first_with_page_number():
    engine, _ = build()
    hits = engine.search("What conditions cause deadlock?", top_k=3)
    assert hits[0]["source"] == "os.pdf" and hits[0]["page"] == 1


def test_non_matching_chunks_get_no_lexical_score():
    # Regression test: with BM25 only (dense_weight=0), chunks sharing no keyword must not be returned.
    engine, _ = build(dense_weight=0.0)
    hits = engine.search("deadlock conditions", top_k=5)
    assert [(h["source"], h["page"]) for h in hits] == [("os.pdf", 1)]


def test_out_of_scope_question_skips_the_chat_model():
    engine, client = build()
    result = engine.ask("What is the capital of France?")
    assert result["answer"] == NO_ANSWER_MESSAGE
    assert result["sources"] == []
    assert client.chat_calls == []


def test_in_scope_question_calls_chat_once_and_returns_citations():
    engine, client = build()
    result = engine.ask("What conditions cause deadlock?", top_k=2)
    assert len(client.chat_calls) == 1
    assert result["sources"][0]["citation"] == "S1"
    assert "[S1]" in client.chat_calls[0]["messages"][1]["content"]


def test_follow_up_is_rewritten_before_retrieval():
    engine, client = build(client=FakeClient(condensed="What conditions cause deadlock?"))
    history = [
        {"role": "user", "content": "Tell me about deadlock"},
        {"role": "assistant", "content": "Deadlock is a state where processes wait forever."},
    ]
    result = engine.ask("and what are its conditions?", history=history)
    assert result["standalone_question"] == "What conditions cause deadlock?"
    assert len(client.chat_calls) == 2  # one rewrite + one answer


def test_min_similarity_can_be_lowered_live():
    engine, _ = build(min_similarity=0.2)
    assert engine.search("zebra giraffe") == []
    engine.min_similarity = 0.0
    assert engine.search("zebra giraffe")  # cutoff disabled: nearest chunks are returned


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        RAGEngine(client=FakeClient(), dense_weight=1.5)
    with pytest.raises(ValueError):
        RAGEngine(client=FakeClient(), min_similarity=-0.1)
    with pytest.raises(ValueError):
        RAGEngine(client=FakeClient(), chunk_size=100, chunk_overlap=100)


def test_asking_before_indexing_fails_clearly():
    engine = RAGEngine(client=FakeClient())
    with pytest.raises(RuntimeError):
        engine.ask("anything")


def test_empty_question_rejected():
    engine, _ = build()
    with pytest.raises(ValueError):
        engine.ask("   ")


def test_missing_api_key_rejected_without_injected_client():
    with pytest.raises(ValueError):
        RAGEngine(api_key="")


def test_real_pdf_extraction_keeps_page_numbers_and_duplicate_names():
    engine = RAGEngine(client=FakeClient())
    pdf = make_pdf(["Deadlock requires four conditions.", "Paging uses page tables."])
    engine.index_pdfs([("notes.pdf", pdf), ("notes.pdf", pdf)])
    assert {c.source for c in engine.chunks} == {"notes.pdf", "notes (2).pdf"}
    hits = engine.search("page tables paging", top_k=1)
    assert hits[0]["page"] == 2


def test_pdf_without_text_is_rejected():
    engine = RAGEngine(client=FakeClient())
    with pytest.raises(ValueError, match="No extractable text"):
        engine.index_pdfs([("blank.pdf", make_pdf([""]))])
