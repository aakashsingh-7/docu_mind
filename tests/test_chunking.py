import pytest

from rag_engine import _split_text, _tokenize


def test_chunks_respect_size_limit():
    text = "Sentence number one is here. " * 200
    chunks = _split_text(text, 1000, 150)
    assert len(chunks) > 1
    assert max(len(c) for c in chunks) <= 1000


def test_empty_text_gives_no_chunks():
    assert _split_text("   \t  ", 1000, 150) == []


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        _split_text("some text", 100, 100)


def test_short_text_is_one_chunk():
    assert _split_text("Short page.", 1000, 150) == ["Short page."]


def test_tokenizer_drops_stop_words():
    assert _tokenize("What is the deadlock condition?") == ["deadlock", "condition"]
