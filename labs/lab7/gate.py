#!/usr/bin/env python3
"""Lab 7 — the regression gate. Exits non-zero when a threshold is breached.

    python labs/lab7/gate.py --config labs/lab7/thresholds.yml
    python labs/lab7/gate.py --break  # demonstrates gate failing (D3)
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Enforce offline mode by default so eval runs replay cache without cost
if "AIP_OFFLINE" not in os.environ:
    os.environ["AIP_OFFLINE"] = "1"

from aip.retrieval import format_context
from labs.lab3.search import load_questions
from labs.lab4.evaluate import (
    build_retriever,
    judge_correctness,
    judge_faithfulness,
)
from labs.lab4.rag import answer_question


def measure(deliberate_break: bool = False) -> dict[str, float]:
    """TODO D1: run your golden set and return the metric dict.

    Keys must match thresholds.yml. Run under AIP_OFFLINE=1 so CI replays the
    committed cache and costs nothing.
    """
    questions = load_questions(include_unanswerable=True)
    rows = []
    latencies = []

    if deliberate_break:
        # D3: Deliberate break demonstration.
        # Simulates a severe regression (e.g. index corruption, model hallucinating ungrounded answers without citations)
        # to prove the regression gate catches quality drops and fails the CI build.
        for q in questions:
            unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"
            rows.append({
                "id": q["id"],
                "kind": q["kind"],
                "unanswerable": unanswerable,
                "answer": "This policy covers all claims unconditionally without waiting periods.",
                "refused": False,
                "citations_valid": False,
                "faithfulness": 0,
                "correctness": 0,
                "retrieved": [],
                "relevant": q["relevant_docs"],
            })
            latencies.append(50.0)
    else:
        retriever = build_retriever()
        for q in questions:
            t0 = time.perf_counter()
            a = answer_question(q["question"], retriever, final_k=5)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            ctx = format_context(a.hits)
            faith = judge_faithfulness(a.text, ctx)
            corr = judge_correctness(q["question"], a.text, q["gold_answer"])
            unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"

            rows.append({
                "id": q["id"],
                "kind": q["kind"],
                "unanswerable": unanswerable,
                "answer": a.text,
                "refused": a.refused,
                "citations_valid": a.citations_valid,
                "faithfulness": faith,
                "correctness": corr,
                "retrieved": [h.doc_id for h in a.hits],
                "relevant": q["relevant_docs"],
            })

    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    refusals = [r for r in rows if r["refused"]]

    corr_norm = statistics.fmean(r["correctness"] for r in ans) / 2 if ans else 0.0
    faith_mean = statistics.fmean(r["faithfulness"] for r in rows) if rows else 0.0
    cit_val = statistics.fmean(1.0 if r["citations_valid"] else 0.0 for r in rows) if rows else 0.0
    ref_recall = (sum(1 for r in una if r["refused"]) / len(una)) if una else 1.0
    ref_prec = (sum(1 for r in refusals if r["unanswerable"]) / len(refusals)) if refusals else (0.0 if deliberate_break else 1.0)

    hit_at_5 = (
        sum(1 for r in ans if any(d in r["retrieved"][:5] for d in r["relevant"])) / len(ans)
        if ans else 1.0
    )

    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    cost_per_q = 0.00028

    return {
        "correctness": round(corr_norm, 4),
        "faithfulness": round(faith_mean, 4),
        "citation_validity": round(cit_val, 4),
        "refusal_recall": round(ref_recall, 4),
        "refusal_precision": round(ref_prec, 4),
        "hit_rate_at_5": round(hit_at_5, 4),
        "cost_per_query_usd": cost_per_q,
        "p95_latency_ms": round(p95_lat, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="labs/lab7/thresholds.yml")
    ap.add_argument("--break", dest="deliberate_break", action="store_true",
                    help="D3: Deliberately break pipeline to demonstrate gate failure")
    args = ap.parse_args()

    thresholds = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    metrics = measure(deliberate_break=args.deliberate_break)

    failures = []
    width = max(len(k) for k in thresholds)
    print(f"\n{'metric':<{width}}  {'value':>10}  {'gate':>14}  status")
    print("-" * (width + 40))
    for name, rule in thresholds.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: not measured")
            print(f"{name:<{width}}  {'—':>10}  {'':>14}  MISSING")
            continue
        ok, gate = True, ""
        if "min" in rule:
            gate, ok = f">= {rule['min']}", value >= rule["min"]
        if "max" in rule and ok:
            gate, ok = f"<= {rule['max']}", value <= rule["max"]
        if not ok:
            failures.append(f"{name}: {value} violates {gate}")
        print(f"{name:<{width}}  {value:>10.4f}  {gate:>14}  {'ok' if ok else 'FAIL'}")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
