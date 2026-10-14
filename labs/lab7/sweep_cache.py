#!/usr/bin/env python3
"""Lab 7 Part B1 — Semantic Cache Threshold Sweep.

Finds the empirical cosine threshold at which semantic caching begins returning
confidently wrong answers to a different question.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.embed import cosine, embed

# Pairs of queries: (base_query, test_query, is_same_intent, reason)
PAIRS = [
    # Positive pairs (True Paraphrases: same question, same answer intended)
    (
        "How many days do I have to submit a reimbursement claim after discharge?",
        "What is the deadline for submitting a reimbursement claim once discharged?",
        True,
        "Identical claim deadline query with synonym phrasing",
    ),
    (
        "What is the room rent limit on the Silver plan?",
        "How much room rent is allowed per day on the Silver plan?",
        True,
        "Same policy parameter asked with per-day clarification",
    ),
    (
        "Is maternity covered on the Bronze plan?",
        "Does the Bronze plan provide coverage for maternity expenses?",
        True,
        "Direct yes/no maternity coverage inquiry on Bronze",
    ),
    (
        "How long is the grace period for an annual policy?",
        "What is the renewal grace period for annual policyholders?",
        True,
        "Same grace period policy parameter",
    ),
    (
        "What is the waiting period for pre-existing diseases?",
        "How many months must I wait before pre-existing conditions are covered?",
        True,
        "PED waiting period inquiry",
    ),

    # Negative pairs (Plan substitutions & near-miss collisions: different answers!)
    (
        "What is the room rent limit on the Silver plan?",
        "What is the room rent limit on the Gold plan?",
        False,
        "Plan substitution: Silver ($5,000) vs Gold (Single Private A/C)",
    ),
    (
        "What is the room rent limit on the Silver plan?",
        "What is the room rent limit on the Platinum plan?",
        False,
        "Plan substitution: Silver vs Platinum (no cap)",
    ),
    (
        "Is maternity covered on the Bronze plan?",
        "Is maternity covered on the Gold plan?",
        False,
        "Plan substitution: Bronze (No) vs Gold (Covered after 24m)",
    ),
    (
        "What is the waiting period for pre-existing diseases?",
        "What is the waiting period for specific diseases?",
        False,
        "Condition difference: PED (36 months) vs Specific Diseases (24 months)",
    ),
    (
        "What is the waiting period for pre-existing diseases?",
        "What is the waiting period for joint replacement surgery?",
        False,
        "Condition difference: General PED vs specific 2-year waiting surgery",
    ),
    (
        "What is the ambulance cover on the Bronze plan?",
        "What is the ambulance cover on the Platinum plan?",
        False,
        "Plan substitution: Bronze (INR 1,500) vs Platinum (Air ambulance covered)",
    ),
]


def run_sweep() -> dict:
    results = []
    print(f"{'Similarity':>10}  {'Label':>12}  {'Base Query':<40}  {'Test Query':<40}")
    print("-" * 110)

    for base_q, test_q, is_same, desc in PAIRS:
        v1 = embed(base_q)
        v2 = embed(test_q)
        sim = float(cosine(v1, v2))
        label = "PARAPHRASE" if is_same else "COLLISION"
        print(f"{sim:>10.4f}  {label:>12}  {base_q[:38]:<40}  {test_q[:38]:<40}")
        results.append({
            "base_q": base_q,
            "test_q": test_q,
            "similarity": sim,
            "is_same": is_same,
            "description": desc,
        })

    thresholds = [0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    print("\n" + "=" * 75)
    print(f"{'Threshold':>10}  {'Paraphrase Hit Rate':>20}  {'Wrong Hit Rate':>18}  {'Status':<15}")
    print("-" * 75)

    pos = [r for r in results if r["is_same"]]
    neg = [r for r in results if not r["is_same"]]

    breaking_threshold = None

    for t in thresholds:
        pos_hits = sum(1 for r in pos if r["similarity"] >= t)
        neg_hits = sum(1 for r in neg if r["similarity"] >= t)
        pos_rate = pos_hits / len(pos)
        neg_rate = neg_hits / len(neg)
        safe = neg_hits == 0

        if not safe and breaking_threshold is None:
            breaking_threshold = t

        status = "SAFE" if safe else f"UNSAFE ({neg_hits} wrong)"
        print(f"{t:>10.2f}  {pos_rate:>19.1%}  {neg_rate:>17.1%}  {status:<15}")

    min_neg_sim = min(r["similarity"] for r in neg)
    max_neg_sim = max(r["similarity"] for r in neg)
    print("\nCollision Analysis:")
    print(f"  Highest collision similarity: {max_neg_sim:.4f}")
    print(f"  Lowest collision similarity:  {min_neg_sim:.4f}")
    print(f"  Safe operational threshold:   >= {max_neg_sim + 0.01:.2f}")

    return {
        "results": results,
        "max_collision_similarity": max_neg_sim,
        "recommended_threshold": round(max_neg_sim + 0.01, 2),
    }


if __name__ == "__main__":
    run_sweep()
