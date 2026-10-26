"""Test helpers: a fake OpenAI-style client and a tiny PDF generator (no network, no API key)."""
import re
import types
import zlib

import numpy as np

from rag_engine import _tokenize

DIM = 4096  # large enough that accidental hash collisions between unrelated words are very unlikely


def fake_embedding(text: str) -> list[float]:
    """Deterministic bag-of-words hashing embedding: shared words => higher cosine similarity."""
    vec = np.zeros(DIM, dtype="float32")
    for token in _tokenize(text):
        vec[zlib.crc32(token.encode()) % DIM] += 1.0
    return vec.tolist()


class FakeClient:
    def __init__(self, condensed: str = "What conditions cause deadlock?"):
        self.chat_calls = []
        self.condensed = condensed
        self.embeddings = types.SimpleNamespace(create=self._embed)
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._chat))

    def _embed(self, model, input):
        return types.SimpleNamespace(
            data=[types.SimpleNamespace(embedding=fake_embedding(t)) for t in input]
        )

    def _chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        system = kwargs["messages"][0]["content"]
        text = self.condensed if system.startswith("Rewrite") else "Answer based on context [S1]."
        message = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


PAGES = [
    ("os.pdf", 1, "Deadlock requires four conditions: mutual exclusion, hold and wait, no preemption and circular wait."),
    ("os.pdf", 2, "Paging divides memory into fixed size frames and pages and uses page tables."),
    ("net.pdf", 5, "TCP uses a three way handshake to establish a connection."),
    ("net.pdf", 6, "Round robin scheduling gives each process a fixed time quantum."),
]


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid text PDF. Page text must not contain parentheses."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
