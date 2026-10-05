#!/usr/bin/env python3
"""Lab 5 — the failure classifier and RAG v2 pipeline.

    python labs/lab5/diagnose.py --input reports/lab4.json
    python labs/lab5/diagnose.py --input reports/lab4.json --pareto --save reports/lab5_diagnosis.json

Implements the T4 §5 diagnostic tree. Everything that can be decided by code
is decided by code; mode 2 needs your eyes and the script says so.
Also includes the Part C fix pipeline (RAG v2 with calibrated prompt and distractor deduplication).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.guards import UNTRUSTED_SYSTEM_CLAUSE  # noqa: E402
from aip.rag import RagPipeline  # noqa: E402
from aip.retrieval import DenseRetriever, Hit  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402

MODES = {
    1: "missing_content",
    2: "chunk_boundary",
    3: "embedding_mismatch",
    4: "ranking",
    5: "reranker",
    6: "generation",
    7: "presentation",
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "but", "not", "what", "which", "who", "whom", "this", "that", "these",
    "those", "then", "so", "than", "too", "very", "can", "will", "just",
    "should", "now", "it", "its", "under", "per", "all", "also", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "only",
    "own", "same", "s", "t", "don"
}


def _clean_tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def answer_in_corpus(gold_answer: str, corpus: dict[str, str],
                     relevant_docs: list[str]) -> bool:
    """Mode 1 test. Normalised entity, numeric, and keyword overlap.

    Unlike the weak baseline substring test, this function:
    1. Handles explicit refusals on unanswerable questions where content is absent.
    2. Validates numerical entities (e.g. monetary amounts, durations, percentages)
       to avoid false positives on questions with specific quantitative requirements.
    3. Normalises tokens with punctuation stripping and stopword filtering,
       preventing false negatives on rephrasings and false positives on generic matches.
    """
    if not relevant_docs:
        return False
    if gold_answer.strip().upper().startswith("REFUSE"):
        return False
    doc_text = " ".join(corpus.get(d, "") for d in relevant_docs).lower()
    if not doc_text:
        return False

    # Extract numerical tokens (stripping formatting commas)
    gold_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", gold_answer.replace(",", "")))
    doc_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", doc_text.replace(",", "")))
    if gold_numbers and not (gold_numbers & doc_numbers):
        return False

    gold_tokens = _clean_tokens(gold_answer)
    if not gold_tokens:
        return True
    doc_tokens = set(_clean_tokens(doc_text))
    overlap = sum(1 for t in gold_tokens if t in doc_tokens) / len(gold_tokens)
    return overlap >= 0.35


def classify(row: dict, q: dict, corpus: dict[str, str], *,
             gold_context_fixes_it: bool | None = None,
             in_top_30: bool | None = None,
             dropped_by_reranker: bool | None = None,
             retriever: DenseRetriever | None = None,
             chunks: list | None = None) -> tuple[int, str]:
    """Walk the T4 §5 diagnostic tree. Returns (mode, evidence).

    Follows the tree in the handout in order:
    1. Mode 7: Right answer, wrong citation
    2. Mode 1: Answer missing from corpus
    3. Mode 6: Generation failure (all gold docs in final k, or fails even with gold context)
    4. Mode 4/5: Ranking / reranker error (gold doc in top 30 but not in final k)
    5. Mode 3: Embedding mismatch (gold doc not in top 30, but retrievable by verbatim text)
    6. Mode 2: Chunk boundary (needs human check)
    """
    # Mode 7 first: right answer, wrong citation.
    if row.get("correctness", 0) >= 2 and not row.get("citations_valid", True):
        return 7, f"correct answer, invalid citations {row.get('invalid_citations')}"

    # Mode 1: is the answer even in the corpus?
    if not answer_in_corpus(q["gold_answer"], corpus, q["relevant_docs"]):
        return 1, "gold answer content not found in the relevant documents"

    retrieved_docs = set(row.get("retrieved", []))
    gold_docs = set(q.get("relevant_docs", []))
    all_gold_in_final = gold_docs.issubset(retrieved_docs) if gold_docs else False

    # Mode 6 check:
    # A. If gold_context_fixes_it is explicitly False, generation was always going to fail.
    if gold_context_fixes_it is False:
        return 6, "generation failed even with gold context"

    # B. If all gold docs were present in the final k context retrieved by the system,
    #    retrieval succeeded. The failure occurred during generation.
    if all_gold_in_final:
        return 6, f"generation failure: all gold docs {list(gold_docs)} were in top-5 context ({list(retrieved_docs)})"

    # Mode 4/5 check: required gold doc was not in final k. Was it in top 30?
    if in_top_30 is None and retriever is not None:
        top30 = retriever.search(q["question"], k=30)
        top30_docs = {h.doc_id for h in top30}
        in_top_30 = bool(gold_docs & top30_docs)

    if in_top_30:
        if dropped_by_reranker:
            return 5, "gold doc in top 30 pool but dropped by reranker"
        missing_docs = list(gold_docs - retrieved_docs)
        return 4, f"ranking failure: gold doc {missing_docs} in top 30 but not in final k ({list(retrieved_docs)})"

    # Mode 3 check: gold doc not in top 30. Check verbatim text retrieval.
    if chunks is not None and retriever is not None:
        gold_chunks = [c for c in chunks if c.doc_id in gold_docs]
        if gold_chunks:
            verb_hits = retriever.search(gold_chunks[0].text[:200], k=10)
            verb_docs = {h.doc_id for h in verb_hits}
            if bool(verb_docs & gold_docs):
                return 3, f"embedding mismatch: gold chunk retrievable by verbatim text but missed by query '{q['question'][:50]}'"

    return 2, "needs_human_check: open the chunks around the gold answer"


def pareto(tally: Counter) -> str:
    total = sum(tally.values()) or 1
    lines, cum = ["failure mode          n    share   cumulative"], 0
    for mode, n in tally.most_common():
        cum += n
        bar = "#" * round(30 * n / total)
        lines.append(f"{MODES[mode]:<20} {n:>3}   {n/total:>5.1%}   "
                     f"{cum/total:>5.1%}  {bar}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Part C — The Fix: RAG v2 Pipeline
# --------------------------------------------------------------------------

ANSWER_SYSTEM_V2 = f"""\
You answer questions using ONLY the numbered sources provided.

Rules, in priority order:
1. If the sources do not contain sufficient information to answer the question, reply exactly:
   "I don't have enough information in the provided sources to answer that."
   Do not guess, and do not fall back on general knowledge.
2. If the sources state that a condition, procedure, or benefit is EXCLUDED or NOT COVERED, state clearly that it is not covered (e.g., "No, [item] is permanently excluded") and cite the source. Do not refuse to answer when the material is explicitly excluded.
3. If the question asks for out-of-pocket costs or calculations based on policy limits stated in the sources, calculate the difference using the figures in the sources and explain the breakdown with citations.
4. If sources contradict each other due to dates or document status (e.g. an archived document vs an active policy), follow the active / latest policy rules (effective 2026) and ignore superseded/archived terms.
5. Every factual sentence must end with a citation of the source(s) that support it, in the form [1] or [2][5].
6. Never cite a number that was not given to you.
7. Be thorough and complete: include all relevant conditions, limits, timelines, and next steps mentioned in the relevant sources.

{UNTRUSTED_SYSTEM_CLAUSE}
"""


class DeduplicatedRagPipeline(RagPipeline):
    """RAG v2: Suppresses duplicate chunks from the same document in top-k context."""
    def rerank(self, question: str, hits: Sequence[Hit]) -> list[Hit]:
        seen_docs = set()
        deduped = []
        for h in hits:
            if h.doc_id not in seen_docs:
                deduped.append(h)
                seen_docs.add(h.doc_id)
            if len(deduped) == self.final_k:
                break
        if len(deduped) < self.final_k:
            for h in hits:
                if h not in deduped:
                    deduped.append(h)
                if len(deduped) == self.final_k:
                    break
        return deduped


def build_v2_pipeline(corpus: dict[str, str], final_k: int = 5) -> DeduplicatedRagPipeline:
    """Build the improved RAG v2 pipeline."""
    chunks = [c for doc_id, text in corpus.items()
              for c in markdown_chunks(text, doc_id, size=800)]
    retriever = DenseRetriever(chunks, show_progress=False)
    return DeduplicatedRagPipeline(retriever, k=15, final_k=final_k, system=ANSWER_SYSTEM_V2)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Lab 5 Failure Diagnosis and Classifier")
    ap.add_argument("--input", default="reports/lab4.json", help="Input evaluation JSON")
    ap.add_argument("--pareto", action="store_true", help="Print Pareto distribution")
    ap.add_argument("--save", default="reports/lab5_diagnosis.json", help="Path to save diagnosis")
    args = ap.parse_args()

    rows = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_questions(include_unanswerable=True)}
    corpus = load_corpus()

    chunks = [c for doc_id, text in corpus.items()
              for c in markdown_chunks(text, doc_id, size=800)]
    retriever = DenseRetriever(chunks, show_progress=False)

    failures = [r for r in rows
                if r.get("correctness", 2) < 2 or not r.get("citations_valid", True)]
    print(f"{len(failures)} failures out of {len(rows)}\n")

    out, tally = [], Counter()
    for r in failures:
        q = questions[r["id"]]
        mode, evidence = classify(r, q, corpus, retriever=retriever, chunks=chunks)
        tally[mode] += 1
        out.append({"id": r["id"], "kind": q["kind"], "mode": mode,
                    "mode_name": MODES[mode], "evidence": evidence,
                    "question": q["question"], "answer": r["answer"][:300]})
        print(f"  {r['id']:<5} {MODES[mode]:<20} {evidence}")

    if args.pareto:
        print("\n" + pareto(tally))

    p = ROOT / args.save
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
