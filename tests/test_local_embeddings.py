import sys
import types
from pathlib import Path

import numpy as np
import pytest

from config import PROVIDERS, settings_from_env
from embedders import LocalEmbedder
from rag_engine import RAGEngine
from tests.helpers import PAGES, FakeClient, fake_embedding


class NoEmbeddingsClient(FakeClient):
    """Chat works, but any call to the embeddings API fails - like Groq, which has no such endpoint."""

    def __init__(self):
        super().__init__()

        def boom(**kwargs):
            raise AssertionError("embeddings API must not be called when a local embedder is set")

        self.embeddings = types.SimpleNamespace(create=boom)


def local_embedder(texts):
    return np.array([fake_embedding(t) for t in texts], dtype="float32")


def test_local_embedder_replaces_embeddings_api():
    client = NoEmbeddingsClient()
    engine = RAGEngine(client=client, embedder=local_embedder)
    engine.index_pages(PAGES)
    hits = engine.search("What conditions cause deadlock?", top_k=2)
    assert hits[0]["source"] == "os.pdf" and hits[0]["page"] == 1


def test_chat_still_goes_through_the_chat_client_with_local_embeddings():
    client = NoEmbeddingsClient()
    engine = RAGEngine(client=client, embedder=local_embedder)
    engine.index_pages(PAGES)
    result = engine.ask("What conditions cause deadlock?", top_k=2)
    assert len(client.chat_calls) == 1
    assert result["sources"]


def test_out_of_scope_still_detected_with_local_embeddings():
    engine = RAGEngine(client=NoEmbeddingsClient(), embedder=local_embedder)
    engine.index_pages(PAGES)
    assert engine.search("What is the capital of France?") == []


def test_missing_fastembed_gives_a_helpful_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)  # makes `import fastembed` raise ImportError
    with pytest.raises(ImportError, match="requirements-local.txt"):
        LocalEmbedder("sentence-transformers/all-MiniLM-L6-v2")


def test_groq_preset_uses_groq_endpoint_and_local_embeddings():
    preset = PROVIDERS["groq"]
    assert preset["base_url"] == "https://api.groq.com/openai/v1"
    assert preset["embedding_backend"] == "local"


def test_groq_settings_from_env(monkeypatch):
    for var in ("LLM_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "CHAT_MODEL", "EMBEDDING_BACKEND"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    cfg = settings_from_env()
    assert cfg["provider"] == "groq"
    assert cfg["api_key"] == "gsk_test"
    assert cfg["embedding_backend"] == "local"
    assert cfg["base_url"] == "https://api.groq.com/openai/v1"


def test_app_accepts_groq_provider_without_errors():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    app = str(Path(__file__).resolve().parents[1] / "app.py")
    at = AppTest.from_file(app, default_timeout=30).run()
    at.selectbox[0].set_value("groq").run()
    assert not at.exception
    assert not any("no embeddings endpoint" in w.value for w in at.sidebar.warning)
