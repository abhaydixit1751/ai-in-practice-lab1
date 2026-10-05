#!/usr/bin/env python3
"""Lab 4 evaluation. Scaffolding provided; the judges are yours.

    python labs/lab4/evaluate.py --full --save reports/lab4.json
    python labs/lab4/evaluate.py --gold-context
    python labs/lab4/evaluate.py --calibrate      # writes the hand-label sheet
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.cost import Budget  # noqa: E402
from aip.evals import (  # noqa: E402
    JUDGE_RUBRIC_CORRECTNESS,
    JUDGE_RUBRIC_FAITHFULNESS,
    judge_agreement,
    llm_judge,
)
from aip.retrieval import DenseRetriever, format_context  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.rag import (  # noqa: E402
    ANSWER_SYSTEM,
    ANSWER_SYSTEM_STRICT,
    REFUSAL,
    answer_question,
    answer_with_gold_context,
)

GOLDEN = ROOT / "data/eval/rag_golden.jsonl"
LABEL_SHEET = ROOT / "labs/lab4/calibration_labels.jsonl"


def build_retriever():
    """Reference configuration: markdown-aware chunking at 400 characters with exact dense retrieval."""
    corpus = load_corpus()
    chunks = [c for doc_id, text in corpus.items()
              for c in markdown_chunks(text, doc_id, size=400)]
    return DenseRetriever(chunks)


# ---------------------------------------------------------------------------
# Improved judges (Part D1)
# ---------------------------------------------------------------------------
IMPROVED_RUBRIC_FAITHFULNESS = """\
You are an expert evaluator grading whether an ANSWER is strictly faithful to and fully supported by the provided CONTEXT.

Evaluation Rules:
1. Support only: Grade ONLY whether the factual claims made in ANSWER are directly substantiated by CONTEXT. Do NOT reward writing style, tone, or external knowledge.
2. World knowledge vs context: Any factual claim NOT supported by CONTEXT must be judged unsupported (score 0), even if it is factually true in the real world.
3. Unsupported extrapolation or strong paraphrasing: If the answer makes a claim stronger than the context justifies (e.g. context says 'may be covered subject to review', but answer says 'guaranteed covered'), that claim is UNSUPPORTED (score 0).
4. Full Refusal: If the answer refuses to answer because context lacks information (e.g., states 'I don't have enough information in the provided sources to answer that'), this is fully FAITHFUL and SUPPORTED (score 1).
5. Partial Refusal: If the answer answers what is supported by context and explicitly declines / states that the remaining requested details are not in the sources, this is FAITHFUL and SUPPORTED (score 1).
6. Citations: Ignore citation brackets like [1], [2] when judging semantic claims; verify that the text claims are backed by the text in CONTEXT.
7. Scoring:
   - Score 1: Every factual claim in ANSWER is directly supported by CONTEXT, or the answer appropriately refuses / partially refuses unsupported claims.
   - Score 0: ANSWER contains at least one factual claim, exaggeration, or number not supported by CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Reply strictly as a valid JSON object: {{"score": 0 or 1, "unsupported_claims": [], "reason": "one sentence explanation"}}
"""

IMPROVED_RUBRIC_CORRECTNESS = """\
Compare a CANDIDATE answer to a REFERENCE gold answer for the specified QUESTION.

Evaluation Rules:
1. Substantive content: Evaluate whether the substantive meaning and key facts match the reference. Exact phrasing does not need to match. Ignore citation tags like [1], [2].
2. Refusal Handling (CRITICAL):
   - If REFERENCE indicates that the question should be refused (e.g. contains 'REFUSE', states the corpus lacks information, or explains no data exists):
     * If CANDIDATE refuses to answer (e.g. states it lacks sufficient information or cannot answer from sources), CANDIDATE is fully correct -> Score 2.
     * If CANDIDATE invents an answer or provides ungrounded facts instead of refusing -> Score 0.
   - If REFERENCE indicates a partial refusal (e.g. 'PARTIAL REFUSE', confirms what is supported but refuses the rest):
     * If CANDIDATE states what is supported AND explicitly notes the missing details cannot be answered -> Score 2.
     * If CANDIDATE gives only the supported facts or only refuses -> Score 1.
     * If CANDIDATE invents the missing details -> Score 0.
   - If REFERENCE provides a factual answer and does NOT call for refusal:
     * If CANDIDATE gives all key facts from reference -> Score 2.
     * If CANDIDATE gives some correct facts but omits a key detail or has minor errors -> Score 1.
     * If CANDIDATE refuses, is factually wrong, or contradicts the reference -> Score 0.

QUESTION: {question}
REFERENCE: {reference}
CANDIDATE: {candidate}

Reply strictly as a valid JSON object: {{"score": 0, 1, or 2, "reason": "one sentence explanation"}}
"""


def judge_faithfulness(answer_text: str, context: str) -> int:
    """D1: Improved faithfulness judge returning 0 or 1.
    Handles full and partial refusals and guards against judge truncation/parse error.
    """
    import time
    prompt = IMPROVED_RUBRIC_FAITHFULNESS.format(
        context=context[:8000], answer=answer_text
    )
    verdict = None
    for attempt in range(5):
        try:
            verdict = llm_judge(prompt, tier="gemini/gemini-3.8-flash")
            break
        except Exception as exc:
            if attempt == 4:
                if REFUSAL[:40].lower() in answer_text.lower():
                    return 1
                return 0
            err_str = str(exc).lower()
            if any(k in err_str for k in ("429", "quota", "rate", "resource_exhausted")):
                time.sleep(20.0)
            else:
                time.sleep(3.0)

    if verdict is None:
        return 1 if REFUSAL[:40].lower() in answer_text.lower() else 0

    if verdict.get("parse_error"):
        try:
            verdict = llm_judge(prompt, tier="gemini/gemini-3.8-flash", max_tokens=4096)
        except Exception:
            pass
    if verdict.get("parse_error") and (REFUSAL[:40].lower() in answer_text.lower()):
        return 1
    return int(verdict.get("score", 0))


def judge_correctness(question: str, candidate: str, reference: str) -> int:
    """D1: Improved correctness judge returning 0, 1, or 2.
    Explicitly handles full and partial refusal cases as specified in the rubric.
    """
    import time
    prompt = IMPROVED_RUBRIC_CORRECTNESS.format(
        question=question, reference=reference, candidate=candidate
    )
    verdict = None
    for attempt in range(5):
        try:
            verdict = llm_judge(prompt, tier="gemini/gemini-3.8-flash")
            break
        except Exception as exc:
            if attempt == 4:
                if "REFUSE" in reference and (REFUSAL[:40].lower() in candidate.lower()):
                    return 2
                return 0
            err_str = str(exc).lower()
            if any(k in err_str for k in ("429", "quota", "rate", "resource_exhausted")):
                time.sleep(20.0)
            else:
                time.sleep(3.0)

    if verdict is None:
        if "REFUSE" in reference and (REFUSAL[:40].lower() in candidate.lower()):
            return 2
        return 0

    if verdict.get("parse_error"):
        try:
            verdict = llm_judge(prompt, tier="gemini/gemini-3.8-flash", max_tokens=4096)
        except Exception:
            pass
    if verdict.get("parse_error"):
        if "REFUSE" in reference and (REFUSAL[:40].lower() in candidate.lower()):
            return 2
    return int(verdict.get("score", 0))


# ---------------------------------------------------------------------------
def run_full(save: str = "", strict: bool = False) -> list[dict]:
    questions = load_questions(include_unanswerable=True)
    retriever = build_retriever()
    rows = []
    system_prompt = ANSWER_SYSTEM_STRICT if strict else ANSWER_SYSTEM

    with Budget(limit_usd=2.00, label="lab4-full") as b:
        for i, q in enumerate(questions, 1):
            a = answer_question(q["question"], retriever, system=system_prompt)
            ctx = format_context(a.hits)
            unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"
            faith = judge_faithfulness(a.text, ctx)
            corr = judge_correctness(q["question"], a.text, q["gold_answer"])
            print(f"[{i:02d}/45 {q['id']}] faith={faith} corr={corr} ref={a.refused}", flush=True)
            rows.append({
                "id": q["id"], "kind": q["kind"], "unanswerable": unanswerable,
                "answer": a.text, "refused": a.refused,
                "citations_valid": a.citations_valid,
                "invalid_citations": a.invalid_citations,
                "faithfulness": faith,
                "correctness": corr,
                "retrieved": [h.doc_id for h in a.hits],
                "relevant": q["relevant_docs"],
            })
            time.sleep(1.0)

    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    refusals = [r for r in rows if r["refused"]]

    print(f"\nn = {len(rows)}  ({len(ans)} answerable, {len(una)} unanswerable)")
    print(f"citation validity   {statistics.fmean(r['citations_valid'] for r in rows):.3f}"
          "   (target 1.000)")
    print(f"faithfulness        {statistics.fmean(r['faithfulness'] for r in rows):.3f}")
    print(f"correctness (0-2)   {statistics.fmean(r['correctness'] for r in ans):.3f}"
          f"  normalised {statistics.fmean(r['correctness'] for r in ans) / 2:.3f}")
    rec = (sum(1 for r in una if r["refused"]) / len(una)) if una else 0.0
    prec = (sum(1 for r in refusals if r["unanswerable"]) / len(refusals)) if refusals else 1.0
    print(f"refusal recall      {rec:.3f}   ({sum(1 for r in una if r['refused'])}/{len(una)})")
    print(f"refusal precision   {prec:.3f}   ({len(refusals)} refusals total)")
    print("\n" + b.report())

    print("\nby question kind (mean correctness / 2):")
    kinds = sorted({r["kind"] for r in ans})
    for kind in kinds:
        sub = [r for r in ans if r["kind"] == kind]
        print(f"  {kind:<16} {statistics.fmean(r['correctness'] for r in sub)/2:.3f}"
              f"  n={len(sub)}")

    if save:
        p = ROOT / save
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved -> {p}   (Lab 5 reads this file)")

    return rows


def run_gold_context() -> None:
    """E2: the decomposition. This is the highest-value 10 minutes in the lab."""
    questions = [q for q in load_questions() if q["relevant_docs"]]
    retriever = build_retriever()
    corpus = load_corpus()

    retrieved_scores, gold_scores = [], []
    with Budget(limit_usd=2.00, label="lab4-decomposition"):
        for q in questions:
            a = answer_question(q["question"], retriever)
            retrieved_scores.append(
                judge_correctness(q["question"], a.text, q["gold_answer"]) / 2)
            g = answer_with_gold_context(
                q["question"], [corpus[d] for d in q["relevant_docs"] if d in corpus])
            gold_scores.append(
                judge_correctness(q["question"], g.text, q["gold_answer"]) / 2)

    A, B = statistics.fmean(gold_scores), statistics.fmean(retrieved_scores)
    print(f"\ncorrectness with GOLD context       A = {A:.3f}   <- generation ceiling")
    print(f"correctness with RETRIEVED context  B = {B:.3f}   <- your system")
    print(f"retrieval-attributable loss   A - B = {A - B:.3f}")
    print(f"generation-attributable loss  1 - A = {1 - A:.3f}")
    print("\nWhichever is larger is where Lab 5 goes.")


def make_calibration_sheet() -> None:
    """D2: writes 20 answers for you to hand-label BEFORE seeing the judge."""
    rows = json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))
    sample = rows[:20]
    LABEL_SHEET.write_text("\n".join(json.dumps({
        "id": r["id"], "answer": r["answer"],
        "human_faithfulness": None, "human_correctness": None,
    }, ensure_ascii=False) for r in sample) + "\n", encoding="utf-8")
    print(f"wrote {LABEL_SHEET}")
    print("Fill in human_faithfulness (0/1) and human_correctness (0/1/2), then:")
    print("  python labs/lab4/evaluate.py --kappa")


def report_kappa() -> None:
    human = [json.loads(l) for l in LABEL_SHEET.open(encoding="utf-8")]
    machine = {r["id"]: r for r in
               json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))}
    for field in ("faithfulness", "correctness"):
        h = [r[f"human_{field}"] for r in human if r[f"human_{field}"] is not None]
        m = [machine[r["id"]][field] for r in human if r[f"human_{field}"] is not None]
        if not h:
            print(f"{field}: no human labels yet")
            continue
        print(f"{field}: {judge_agreement(m, h)}")
    print("\nkappa < 0.4 -> fix the rubric, not the model. Read your disagreements.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--strict", action="store_true", help="Run with strict refusal prompt")
    ap.add_argument("--gold-context", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--kappa", action="store_true")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    if a.full:
        run_full(a.save, strict=a.strict)
    if a.gold_context:
        run_gold_context()
    if a.calibrate:
        make_calibration_sheet()
    if a.kappa:
        report_kappa()
    if not any([a.full, a.gold_context, a.calibrate, a.kappa]):
        ap.print_help()
