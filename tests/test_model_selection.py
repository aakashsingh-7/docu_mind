from pathlib import Path

import httpx
import pytest
from openai import RateLimitError

import rag_engine
from config import PROVIDERS, filter_chat_models, pick_best_chat_model

GROQ_LIKE = [
    "whisper-large-v3", "canopylabs/orpheus-v1-english", "groq/compound", "meta-llama/llama-guard-4-12b",
    "openai/gpt-oss-safeguard-20b", "llama-3.1-8b-instant", "openai/gpt-oss-20b", "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
]


def test_filter_removes_speech_safety_and_agentic_models():
    kept = filter_chat_models(GROQ_LIKE)
    assert "openai/gpt-oss-120b" in kept and "llama-3.1-8b-instant" in kept
    for bad in ("whisper-large-v3", "groq/compound", "meta-llama/llama-guard-4-12b", "openai/gpt-oss-safeguard-20b"):
        assert bad not in kept


def test_best_model_is_gpt_oss_120b_when_available():
    assert pick_best_chat_model("groq", filter_chat_models(GROQ_LIKE)) == "openai/gpt-oss-120b"


def test_falls_back_to_next_preference():
    assert pick_best_chat_model("groq", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]) == "llama-3.3-70b-versatile"


def test_falls_back_to_any_model_then_default():
    assert pick_best_chat_model("groq", ["some/new-model"]) == "some/new-model"
    assert pick_best_chat_model("groq", [], fallback="x") == "x"


def test_groq_default_model_is_the_preferred_one():
    assert PROVIDERS["groq"]["chat_model"] == "openai/gpt-oss-120b"


def _rate_limit_error(retry_after=None):
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "http://example.test"))
    return RateLimitError("rate limited", response=response, body=None)


def test_retry_honours_retry_after_header(monkeypatch):
    sleeps = []
    monkeypatch.setattr(rag_engine.time, "sleep", sleeps.append)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _rate_limit_error("17")
        return "ok"

    assert rag_engine._with_retry(flaky) == "ok"
    assert sleeps == [17.0]


def test_retry_gives_up_after_all_attempts(monkeypatch):
    monkeypatch.setattr(rag_engine.time, "sleep", lambda s: None)

    def always():
        raise _rate_limit_error()

    with pytest.raises(RateLimitError):
        rag_engine._with_retry(always, attempts=3)


def test_app_load_models_button_selects_best_model(monkeypatch):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    monkeypatch.setattr("config.fetch_model_ids", lambda key, url: GROQ_LIKE)
    app = str(Path(__file__).resolve().parents[1] / "app.py")
    at = AppTest.from_file(app, default_timeout=30).run()
    at.selectbox[0].set_value("groq").run()
    at.sidebar.text_input[0].set_value("gsk_fake_key_for_test").run()
    next(b for b in at.sidebar.button if b.label == "Load models from provider").click().run()
    assert not at.exception
    chat = next(s for s in at.sidebar.selectbox if s.label == "Chat model")
    assert chat.value == "openai/gpt-oss-120b"
    assert "whisper-large-v3" not in chat.options
