"""Provider presets and environment-based settings.

DocuMind talks to any OpenAI-compatible API through the official `openai` SDK.
Model names change over time: if a provider rejects a default below, check the
provider's current model list and override it in the sidebar or in `.env`.
"""
from __future__ import annotations

import os

from embedders import LOCAL_EMBEDDING_MODEL

PROVIDERS: dict[str, dict] = {
    "openai": {
        "label": "OpenAI",
        "base_url": None,
        "chat_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "embedding_backend": "api",
    },
    "gemini": {
        "label": "Google Gemini (OpenAI-compatible endpoint)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "chat_model": "gemini-2.0-flash",
        "embedding_model": "gemini-embedding-001",
        "embedding_backend": "api",
    },
    "groq": {
        # Groq serves chat models only (no embeddings endpoint), so embeddings run locally and free.
        "label": "Groq (chat) + free local embeddings",
        "base_url": "https://api.groq.com/openai/v1",
        "chat_model": "openai/gpt-oss-120b",
        "embedding_model": "",
        "embedding_backend": "local",
    },
    "custom": {
        "label": "Custom OpenAI-compatible endpoint",
        "base_url": None,
        "chat_model": "",
        "embedding_model": "",
        "embedding_backend": "api",
    },
}


def env_provider() -> str:
    value = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    return value if value in PROVIDERS else "openai"


def env_api_key() -> str:
    return (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or ""
    )


def settings_from_env() -> dict:
    """Connection settings for command-line tools such as evaluate.py."""
    provider = env_provider()
    preset = PROVIDERS[provider]
    return {
        "provider": provider,
        "api_key": env_api_key(),
        "base_url": os.getenv("LLM_BASE_URL") or preset["base_url"],
        "chat_model": os.getenv("CHAT_MODEL") or preset["chat_model"],
        "embedding_model": os.getenv("EMBEDDING_MODEL") or preset["embedding_model"],
        "embedding_backend": (os.getenv("EMBEDDING_BACKEND") or preset["embedding_backend"]).strip().lower(),
        "local_embedding_model": os.getenv("LOCAL_EMBEDDING_MODEL") or LOCAL_EMBEDDING_MODEL,
    }


# Best-first preference when picking a chat model automatically from what the account can use.
# Groq's own docs list openai/gpt-oss-120b as its largest production chat model.
CHAT_MODEL_PREFERENCE: dict[str, list[str]] = {
    "groq": [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
    ],
}

# Model ids containing these are speech, safety or agentic systems, not plain chat models for grounded RAG.
_NON_CHAT_MARKERS = ("whisper", "tts", "orpheus", "guard", "safeguard", "embed", "compound")


def filter_chat_models(model_ids: list[str]) -> list[str]:
    return sorted(m for m in model_ids if not any(x in m.lower() for x in _NON_CHAT_MARKERS))


def pick_best_chat_model(provider: str, available: list[str], fallback: str = "") -> str:
    """First preferred model the account can use, else the first available one, else `fallback`."""
    for preferred in CHAT_MODEL_PREFERENCE.get(provider, []):
        if preferred in available:
            return preferred
    return available[0] if available else fallback


def fetch_model_ids(api_key: str, base_url: str | None) -> list[str]:
    """Ask the provider which models this key can use (OpenAI-compatible /models endpoint)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    return [m.id for m in client.models.list().data]
