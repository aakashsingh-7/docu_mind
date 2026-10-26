"""Measure retrieval quality on YOUR OWN test questions.

Usage:
    python evaluate.py --pdfs path/to/pdf_folder --questions eval/questions.csv
    python evaluate.py --pdfs docs_folder --questions eval/questions.csv --with-answers

questions.csv columns:
    question          the question text
    expected_source   PDF file name that contains the answer (leave empty if answerable = no)
    expected_pages    page numbers holding the answer, separated by ; (for example 12;13)
    answerable        yes / no  (no = the answer is NOT in your documents; default yes)

The script prints counts (not invented percentages) and writes a CSV you can paste into your report.
Rows whose question starts with "EXAMPLE" are skipped.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from config import settings_from_env
from embedders import LocalEmbedder
from rag_engine import RAGEngine


def parse_pages(value: str) -> set[int]:
    return {int(p) for p in re.split(r"[;,\s]+", (value or "").strip()) if p.isdigit()}


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdfs", required=True, help="Folder containing the PDFs to index")
    parser.add_argument("--questions", required=True, help="CSV file of test questions")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--dense-weight", type=float, default=0.65)
    parser.add_argument("--min-similarity", type=float, default=0.2)
    parser.add_argument("--with-answers", action="store_true", help="Also generate answers (uses the chat model)")
    parser.add_argument("--out", default="eval/results.csv")
    args = parser.parse_args()

    cfg = settings_from_env()
    if not cfg["api_key"]:
        print("Set LLM_API_KEY in .env first.", file=sys.stderr)
        return 1

    pdf_paths = sorted(Path(args.pdfs).glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {args.pdfs}", file=sys.stderr)
        return 1

    embedder = LocalEmbedder(cfg["local_embedding_model"]) if cfg["embedding_backend"] == "local" else None
    engine = RAGEngine(
        embedder=embedder,
        api_key=cfg["api_key"],
        model=cfg["chat_model"],
        embedding_model=cfg["embedding_model"],
        base_url=cfg["base_url"],
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        dense_weight=args.dense_weight,
        min_similarity=args.min_similarity,
    )
    print(f"Indexing {len(pdf_paths)} PDF(s)…")
    engine.index_pdfs([(p.name, p.read_bytes()) for p in pdf_paths])
    print(f"Chunks: {len(engine.chunks)}")

    with open(args.questions, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("question") and not r["question"].startswith("EXAMPLE")]
    if not rows:
        print("No usable questions found in the CSV.", file=sys.stderr)
        return 1

    results = []
    answerable_n = hits = abstain_n = abstained = 0
    reciprocal_ranks = []
    for row in rows:
        question = row["question"].strip()
        answerable = (row.get("answerable") or "yes").strip().lower() != "no"
        expected_source = (row.get("expected_source") or "").strip()
        expected_pages = parse_pages(row.get("expected_pages", ""))
        found = engine.search(question, top_k=args.top_k)
        retrieved = [f"{h['source']}:p{h['page']}" for h in found]

        hit = ""
        rank = ""
        did_abstain = ""
        if answerable:
            answerable_n += 1
            for h in found:
                if h["source"] == expected_source and h["page"] in expected_pages:
                    hit, rank = "yes", h["rank"]
                    break
            else:
                hit = "no"
            hits += hit == "yes"
            reciprocal_ranks.append(1 / rank if rank else 0.0)
        else:
            abstain_n += 1
            did_abstain = "yes" if not found else "no"
            abstained += did_abstain == "yes"

        answer = ""
        if args.with_answers:
            answer = engine.ask(question, top_k=args.top_k)["answer"]

        results.append({
            "question": question,
            "answerable": "yes" if answerable else "no",
            "expected_source": expected_source,
            "expected_pages": row.get("expected_pages", ""),
            "retrieved": " | ".join(retrieved),
            "hit_at_k": hit,
            "rank_of_first_hit": rank,
            "abstained": did_abstain,
            "answer": answer,
            "manual_answer_correct": "",   # fill in by hand: yes / no / partial
            "manual_faithful_to_sources": "",  # fill in by hand: yes / no
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print("\n=== Summary (report these exact numbers, with the settings below) ===")
    print(f"Documents: {len(pdf_paths)} | chunks: {len(engine.chunks)} | questions: {len(rows)}")
    print(f"Settings: chunk={args.chunk_size}, overlap={args.overlap}, top_k={args.top_k}, "
          f"dense_weight={args.dense_weight}, min_similarity={args.min_similarity}")
    if answerable_n:
        mrr = sum(reciprocal_ranks) / answerable_n
        print(f"Answerable questions with the correct page in top-{args.top_k}: {hits} of {answerable_n} "
              f"(mean reciprocal rank {mrr:.3f})")
    if abstain_n:
        print(f"Unanswerable questions correctly returning no passages: {abstained} of {abstain_n}")
    print(f"Per-question results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
