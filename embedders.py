"""Free, local embeddings (no API key, no network after the first model download).

Needed when your chat provider has no embeddings endpoint (for example Groq).
Install once:  pip install -r requirements-local.txt
"""
from __future__ import annotations

import numpy as np

LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbedder:
    """Callable that turns a list of strings into a float32 matrix, using fastembed (ONNX, CPU)."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on the user's environment
            raise ImportError(
                "Local embeddings need the optional package 'fastembed'. "
                "Run: pip install -r requirements-local.txt"
            ) from exc
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def __call__(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self._model.embed(texts)), dtype="float32")
