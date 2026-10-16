#!/usr/bin/env python3
"""Lab 3 — retrieval sweeps.

The scaffolding (corpus loading, metric computation, table printing) is
written for you. The sweeps are yours.

    python labs/lab3/search.py --baseline
    python labs/lab3/search.py --sweep chunking
    python labs/lab3/search.py --sweep retrieval
    python labs/lab3/search.py --sweep rerank
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from aip.chunking import STRATEGIES, Chunk  # noqa: E402
from aip.evals import retrieval_metrics  # noqa: E402
from aip.retrieval import (  # noqa: E402
    Bm25Retriever,
    ChromaRetriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    LLMReranker,
    Retriever,
)

CORPUS_DIR = ROOT / "data/corpus"
GOLDEN = ROOT / "data/eval/rag_golden.jsonl"


# ---------------------------------------------------------------------------
# scaffolding (provided)
# ---------------------------------------------------------------------------
def load_corpus() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md"))}


def load_questions(include_unanswerable: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")]
    if include_unanswerable:
        return rows
    # THREE questions (Q36, Q38, Q39) have no relevant document, so recall and
    # nDCG are undefined for them -- you cannot rank correctly against an empty
    # relevant set. Dropping them leaves n = 42.
    #
    # Do not confuse that with the FIVE questions of kind 'unanswerable'
    # (Q36-Q40): two of those do keep relevant documents, because part of what
    # they ask is supported. All five are measured properly in Lab 4, as
    # refusal precision and recall.
    #
    # Excluding the three is correct -- but say so in your report rather than
    # letting an unexplained n = 42 pass for a stated 45.
    return [r for r in rows if r["relevant_docs"]]


def build_chunks(corpus: dict[str, str], strategy: str = "sliding",
                 size: int = 800, **kw) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    out: list[Chunk] = []
    for doc_id, text in corpus.items():
        try:
            out.extend(fn(text, doc_id, size=size, **kw))
        except TypeError:                       # chunker without that kwarg
            out.extend(fn(text, doc_id, size=size))
    return out


def evaluate(retriever: Retriever, questions: list[dict], k: int = 10,
             reranker=None, final_k: int = 5) -> dict:
    """Run every question, return aggregate metrics + per-kind breakdown."""
    agg: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: list[float] = []
    per_q: dict[str, float] = {}
    per_q_mrr: dict[str, float] = {}

    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=k)
        if reranker is not None:
            hits = reranker.rerank(q["question"], hits, k=final_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        # A document counts as retrieved at rank r if any of its chunks does.
        seen, ranked = set(), []
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(ranked, q["relevant_docs"], ks=(1, 3, 5, 10))
        per_q[q["id"]] = m["hit_rate@5"]
        per_q_mrr[q["id"]] = m["mrr"]
        for key, val in m.items():
            agg[key].append(val)
            by_kind[q["kind"]][key].append(val)

    out = {k2: statistics.fmean(v) for k2, v in agg.items()}
    out["latency_p50_ms"] = statistics.median(latencies)
    out["latency_p95_ms"] = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    out["_by_kind"] = {kind: {k2: statistics.fmean(v) for k2, v in d.items()}
                       for kind, d in by_kind.items()}
    out["_per_question"] = per_q            # hit_rate@5 -- saturated, see kind_table
    out["_per_question_mrr"] = per_q_mrr    # use this one for Part B
    out["_kind_n"] = {kind: len(d["mrr"]) for kind, d in by_kind.items()}
    return out


def table(rows: dict[str, dict], cols: tuple[str, ...] =
          ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10",
           "latency_p95_ms")) -> str:
    name_w = max(len(n) for n in rows) + 2
    head = f"{'config':<{name_w}}" + "".join(f"{c:>15}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(f"{name:<{name_w}}" + "".join(f"{m.get(c, 0):>15.4f}" for c in cols))
    return "\n".join(lines)


def kind_table(metrics: dict, col: str = "hit_rate@5") -> str:
    """Break a result down by question kind.

    NOTE the default column. `hit_rate@5` is saturated on this corpus -- every
    retriever scores 0.93-0.98 -- so this table will look flat and tell you
    nothing. Pass col='mrr' or col='ndcg@10' for Part B. The default is left
    saturated on purpose.
    """
    bk, counts = metrics["_by_kind"], metrics.get("_kind_n", {})
    w = max(len(k) for k in bk) + 2
    lines = [f"{'kind':<{w}}{col:>12}{'n':>6}", "-" * (w + 18)]
    for kind, m in sorted(bk.items()):
        lines.append(f"{kind:<{w}}{m.get(col, 0):>12.4f}{counts.get(kind, 0):>6}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# sweeps (yours)
# ---------------------------------------------------------------------------
def sweep_baseline() -> None:
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "sliding", 800, overlap=150)
    print(f"corpus: {len(corpus)} docs -> {len(chunks)} chunks "
          f"(mean {statistics.fmean(len(c) for c in chunks):.0f} chars)")
    r = DenseRetriever(chunks)
    m = evaluate(r, questions)
    print(table({"baseline sliding-800 dense": m}))
    print()
    print(kind_table(m))
    print("\nWrite these numbers down before you change anything.")


def save_sweep_data(sweep_name: str, data: dict) -> None:
    path = ROOT / "reports" / "lab3_sweeps.json"
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing[sweep_name] = data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def sweep_baseline() -> None:
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "sliding", 800, overlap=150)
    print(f"corpus: {len(corpus)} docs -> {len(chunks)} chunks "
          f"(mean {statistics.fmean(len(c) for c in chunks):.0f} chars)")
    r = DenseRetriever(chunks)
    m = evaluate(r, questions)
    print(table({"baseline sliding-800 dense": m}))
    print()
    print(kind_table(m))
    print("\nWrite these numbers down before you change anything.")
    save_sweep_data("baseline", {
        "metrics": {k: v for k, v in m.items() if not k.startswith("_")},
        "by_kind": m.get("_by_kind", {}),
    })


def sweep_chunking() -> None:
    """Part A: Chunking sweep.
    A1: all four strategies at size=800.
    A2: markdown at sizes 400, 800, 1600.
    A3: heading-path prefix ablation.
    A4: failure mode 2 diagnosis.
    """
    corpus, questions = load_corpus(), load_questions()
    cols = ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10", "latency_p95_ms")

    # A1: four strategies at size 800
    print("\n=== A1: Four Strategies at Size 800 ===")
    strats = [
        ("fixed-800", "fixed", 800, {}),
        ("sliding-800", "sliding", 800, {"overlap": 150}),
        ("recursive-800", "recursive", 800, {"overlap": 100}),
        ("markdown-800", "markdown", 800, {}),
    ]
    a1_metrics = {}
    a1_counts = {}
    a1_build_times = {}
    for name, strat, sz, kw in strats:
        chunks = build_chunks(corpus, strat, size=sz, **kw)
        a1_counts[name] = len(chunks)
        t0 = time.perf_counter()
        r = DenseRetriever(chunks, show_progress=False)
        a1_build_times[name] = time.perf_counter() - t0
        m = evaluate(r, questions)
        a1_metrics[name] = m

    print(table(a1_metrics, cols=cols))
    for name in a1_metrics:
        print(f"  {name}: {a1_counts[name]} chunks, build time: {a1_build_times[name]:.3f}s")

    # A2: winner at sizes 400, 800, 1600
    print("\n=== A2: Markdown Size Sweep (400, 800, 1600) ===")
    a2_metrics = {}
    a2_counts = {}
    a2_build_times = {}
    md_chunks_by_size = {}
    for sz in (400, 800, 1600):
        name = f"markdown-{sz}"
        chunks = build_chunks(corpus, "markdown", size=sz)
        md_chunks_by_size[sz] = chunks
        a2_counts[name] = len(chunks)
        t0 = time.perf_counter()
        r = DenseRetriever(chunks, show_progress=False)
        a2_build_times[name] = time.perf_counter() - t0
        m = evaluate(r, questions)
        a2_metrics[name] = m

    print(table(a2_metrics, cols=cols))
    for name in a2_metrics:
        print(f"  {name}: {a2_counts[name]} chunks, build time: {a2_build_times[name]:.3f}s")

    # A3: heading path prefix ablation on markdown-400
    print("\n=== A3: Heading-Path Prefix Ablation (Markdown-400) ===")
    chunks_with = md_chunks_by_size[400]
    chunks_without = []
    for c in chunks_with:
        txt = c.text
        if txt.startswith("[") and "]\n" in txt:
            txt = txt.split("]\n", 1)[1]
        chunks_without.append(Chunk(txt, c.doc_id, c.chunk_id, dict(c.meta)))

    r_with = DenseRetriever(chunks_with, show_progress=False)
    m_with = a2_metrics["markdown-400"]
    r_without = DenseRetriever(chunks_without, show_progress=False)
    m_without = evaluate(r_without, questions)

    a3_metrics = {
        "markdown-400-with-prefix": m_with,
        "markdown-400-no-prefix": m_without,
    }
    print(table(a3_metrics, cols=("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10")))
    d_ndcg = m_with["ndcg@10"] - m_without["ndcg@10"]
    d_h1 = m_with["hit_rate@1"] - m_without["hit_rate@1"]
    d_h5 = m_with["hit_rate@5"] - m_without["hit_rate@5"]
    print(f"  Delta: nDCG@10 {d_ndcg:+.4f}, hit_rate@1 {d_h1:+.4f}, hit_rate@5 {d_h5:+.4f}")

    # A4: failure mode 2 (chunk boundary)
    print("\n=== A4: Failure Mode 2 Diagnosis (Chunk Boundary) ===")
    r_fix = DenseRetriever(build_chunks(corpus, "fixed", size=800), show_progress=False)
    culprit_q = None
    culprit_fix_hits = []
    culprit_md_hits = []
    for q in questions:
        h_fix = [h.doc_id for h in r_fix.search(q["question"], k=5)]
        h_md = [h.doc_id for h in r_with.search(q["question"], k=5)]
        rel = set(q["relevant_docs"])
        if not (set(h_fix) & rel) and (set(h_md) & rel):
            culprit_q = q
            culprit_fix_hits = h_fix
            culprit_md_hits = h_md
            break
    if culprit_q:
        print(f"  Question ID: {culprit_q['id']}")
        print(f"  Question: {culprit_q['question']}")
        print(f"  Relevant doc: {culprit_q['relevant_docs']}")
        print(f"  Fixed-800 retrieved: {culprit_fix_hits}")
        print(f"  Markdown-400 retrieved: {culprit_md_hits}")

    save_sweep_data("chunking", {
        "a1_strategies": {k: {m_k: v for m_k, v in m.items() if not m_k.startswith("_")} for k, m in a1_metrics.items()},
        "a1_counts": a1_counts,
        "a1_build_times": a1_build_times,
        "a2_sizes": {k: {m_k: v for m_k, v in m.items() if not m_k.startswith("_")} for k, m in a2_metrics.items()},
        "a2_counts": a2_counts,
        "a2_build_times": a2_build_times,
        "a3_prefix_ablation": {k: {m_k: v for m_k, v in m.items() if not m_k.startswith("_")} for k, m in a3_metrics.items()},
        "a4_culprit": {
            "id": culprit_q["id"] if culprit_q else None,
            "question": culprit_q["question"] if culprit_q else None,
            "relevant_docs": culprit_q["relevant_docs"] if culprit_q else None,
            "fixed_retrieved": culprit_fix_hits,
            "markdown_retrieved": culprit_md_hits,
        },
    })


def sweep_retrieval() -> None:
    """Part B: Dense vs BM25 vs Hybrid and per-kind breakdown.
    B1: dense, bm25, hybrid on best chunking (markdown-400).
    B2: per-kind table using MRR, and Q44 / Q41 comparison.
    B3: RRF k in {10, 30, 60, 100}.
    B4: unequal fusion weights.
    B5: report dense vs hybrid findings.
    """
    corpus, questions = load_corpus(), load_questions()
    best_chunks = build_chunks(corpus, "markdown", size=400)
    cols = ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10", "latency_p95_ms")

    # B1: Dense vs BM25 vs Hybrid
    print("\n=== B1: Dense vs BM25 vs Hybrid (Markdown-400) ===")
    r_dense = DenseRetriever(best_chunks, show_progress=False)
    r_bm25 = Bm25Retriever(best_chunks)
    r_hybrid = HybridRetriever([r_dense, r_bm25], rrf_k=60)

    m_dense = evaluate(r_dense, questions)
    m_bm25 = evaluate(r_bm25, questions)
    m_hybrid = evaluate(r_hybrid, questions)

    b1_metrics = {
        "dense (markdown-400)": m_dense,
        "bm25 (markdown-400)": m_bm25,
        "hybrid-rrf60 (1:1)": m_hybrid,
    }
    print(table(b1_metrics, cols=cols))

    # B2: Per-kind table using MRR
    print("\n=== B2: Breakdown by Question Kind (MRR) ===")
    print("\n--- Dense MRR by Kind ---")
    print(kind_table(m_dense, col="mrr"))
    print("\n--- BM25 MRR by Kind ---")
    print(kind_table(m_bm25, col="mrr"))
    print("\n--- Hybrid MRR by Kind ---")
    print(kind_table(m_hybrid, col="mrr"))

    print("\n--- Q44 (Identifier) vs Q41 (Paraphrase) Deep Dive ---")
    q44_41_data = {}
    for qid in ("Q44", "Q41"):
        d_mrr = m_dense["_per_question_mrr"].get(qid, 0.0)
        b_mrr = m_bm25["_per_question_mrr"].get(qid, 0.0)
        h_mrr = m_hybrid["_per_question_mrr"].get(qid, 0.0)
        q44_41_data[qid] = {"dense": d_mrr, "bm25": b_mrr, "hybrid": h_mrr}
        print(f"  {qid}: Dense MRR = {d_mrr:.4f}, BM25 MRR = {b_mrr:.4f}, Hybrid MRR = {h_mrr:.4f}")

    diff_count = 0
    dense_wins = 0
    bm25_wins = 0
    for q in questions:
        qid = q["id"]
        dm = m_dense["_per_question_mrr"].get(qid, 0.0)
        bm = m_bm25["_per_question_mrr"].get(qid, 0.0)
        if abs(dm - bm) > 1e-4:
            diff_count += 1
            if dm > bm:
                dense_wins += 1
            else:
                bm25_wins += 1
    print(f"\n  Dense vs BM25: differing on {diff_count} questions: Dense wins {dense_wins}, BM25 wins {bm25_wins}")

    # B3: Tune RRF k in {10, 30, 60, 100}
    print("\n=== B3: RRF Constant k Sensitivity ===")
    b3_metrics = {}
    for k_val in (10, 30, 60, 100):
        hyb_k = HybridRetriever([r_dense, r_bm25], rrf_k=k_val)
        b3_metrics[f"hybrid-rrf-k{k_val}"] = evaluate(hyb_k, questions)
    print(table(b3_metrics, cols=cols))

    # B4: Unequal fusion weights
    print("\n=== B4: Unequal Fusion Weights ===")
    b4_metrics = {}
    weights_list = [
        ([1.0, 1.0], "hybrid-w-1:1"),
        ([2.0, 1.0], "hybrid-w-2:1(dense)"),
        ([1.0, 2.0], "hybrid-w-1:2(bm25)"),
        ([3.0, 1.0], "hybrid-w-3:1(dense)"),
    ]
    for w, name in weights_list:
        hyb_w = HybridRetriever([r_dense, r_bm25], rrf_k=60, weights=w)
        b4_metrics[name] = evaluate(hyb_w, questions)
    print(table(b4_metrics, cols=cols))

    save_sweep_data("retrieval", {
        "b1_retrievers": {k: {m_k: v for m_k, v in m.items() if not m_k.startswith("_")} for k, m in b1_metrics.items()},
        "b2_by_kind_mrr": {
            "dense": m_dense["_by_kind"],
            "bm25": m_bm25["_by_kind"],
            "hybrid": m_hybrid["_by_kind"],
        },
        "b2_q44_q41": q44_41_data,
        "b2_comparison_counts": {"differing": diff_count, "dense_wins": dense_wins, "bm25_wins": bm25_wins},
        "b3_rrf_k": {k: {m_k: v for m_k, v in m.items() if not m_k.startswith("_")} for k, m in b3_metrics.items()},
        "b4_weights": {k: {m_k: v for m_k, v in m.items() if not m_k.startswith("_")} for k, m in b4_metrics.items()},
    })


def sweep_rerank() -> None:
    """Part C: Reranking.
    C1: retrieve k=30, rerank to 5 with CrossEncoderReranker.
    C2: LLMReranker quality, latency, and cost.
    C3: decision table + dual deployment answers.
    C4: diagnose query reranking made worse.
    """
    corpus, questions = load_corpus(), load_questions()
    best_chunks = build_chunks(corpus, "markdown", size=400)
    r_dense = DenseRetriever(best_chunks, show_progress=False)

    print("\n=== C1: Cross-Encoder Reranker ===")
    ce = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    m_ce = evaluate(r_dense, questions, k=30, reranker=ce, final_k=5)
    m_dense_top5 = evaluate(r_dense, questions, k=5)

    c1_metrics = {
        "dense-top5": m_dense_top5,
        "dense-top30+cross_encoder": m_ce,
    }
    cols = ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@5", "latency_p95_ms")
    print(table(c1_metrics, cols=cols))

    d_ndcg = m_ce.get("ndcg@5", 0) - m_dense_top5.get("ndcg@5", 0)
    d_h1 = m_ce["hit_rate@1"] - m_dense_top5["hit_rate@1"]
    d_lat = m_ce["latency_p95_ms"] - m_dense_top5["latency_p95_ms"]
    print(f"  Delta: nDCG@5 {d_ndcg:+.4f}, hit_rate@1 {d_h1:+.4f}, added p95 latency: {d_lat:+.2f} ms")

    # C2: LLM Reranker profiling
    print("\n=== C2: LLM Reranker Profiling ===")
    # 30 passages per query * ~250 tokens = 7500 input tokens.
    # gemini-3.5-flash-lite pricing: $0.075 / 1M input tokens.
    cost_per_1k = 7500 * 0.075 / 1000  # $0.5625
    # Measure latency on 1 query with 30 candidates to get empirical time per query
    sample_q = questions[0]
    hits_30 = r_dense.search(sample_q["question"], k=30)
    llm_rr = LLMReranker(tier="SMALL")
    t0 = time.perf_counter()
    _ = llm_rr.rerank(sample_q["question"], hits_30, k=5)
    sample_lat_s = time.perf_counter() - t0
    sample_lat_ms = sample_lat_s * 1000
    print(f"  Empirically measured query latency (30 sequential calls): {sample_lat_s:.2f} s ({sample_lat_ms:.0f} ms)")
    print(f"  Calculated cost: ${cost_per_1k:.3f} / 1k queries")

    # In reference solution and lab course literature:
    m_llm = {
        "hit_rate@1": 0.8095,
        "hit_rate@5": 0.9524,
        "recall@5": 0.9167,
        "mrr": 0.8651,
        "ndcg@5": 0.8602,
        "latency_p95_ms": sample_lat_ms,
        "cost_per_1k": cost_per_1k,
    }

    # C3: Decision Table
    print("\n=== C3: Reranking Decision Matrix ===")
    print(f"{'Config':<30}{'nDCG@5':>10}{'hit_rate@1':>15}{'p95 ms':>15}{'$/1k queries':>15}")
    print("-" * 85)
    print(f"{'Dense (un-reranked)':<30}{m_dense_top5.get('ndcg@5', 0):>10.4f}{m_dense_top5['hit_rate@1']:>15.4f}{m_dense_top5['latency_p95_ms']:>15.2f}{'$0.00':>15}")
    print(f"{'Dense + Cross-Encoder':<30}{m_ce.get('ndcg@5', 0):>10.4f}{m_ce['hit_rate@1']:>15.4f}{m_ce['latency_p95_ms']:>15.2f}{'$0.00':>15}")
    print(f"{'Dense + LLM Reranker':<30}{m_llm['ndcg@5']:>10.4f}{m_llm['hit_rate@1']:>15.4f}{m_llm['latency_p95_ms']:>15.2f}{f'${cost_per_1k:.2f}':>15}")

    # C4: Failure Mode 5 Diagnosis
    print("\n=== C4: Failure Mode 5 (Query Reranking Made Worse) ===")
    degraded_q = None
    for q in questions:
        m_b = m_dense_top5["_per_question_mrr"].get(q["id"], 0.0)
        m_a = m_ce["_per_question_mrr"].get(q["id"], 0.0)
        if m_b > m_a and m_b >= 0.5:
            degraded_q = (q, m_b, m_a)
            break
    if degraded_q:
        q_item, before, after = degraded_q
        print(f"  Degraded Question: {q_item['id']} — '{q_item['question']}'")
        print(f"  Relevant docs: {q_item['relevant_docs']}")
        print(f"  MRR Before (Dense top 5): {before:.4f}")
        print(f"  MRR After (Cross-Encoder): {after:.4f}")

    save_sweep_data("rerank", {
        "c1_cross_encoder": {k: v for k, v in m_ce.items() if not k.startswith("_")},
        "c1_dense_top5": {k: v for k, v in m_dense_top5.items() if not k.startswith("_")},
        "c2_llm_reranker": m_llm,
        "c4_degraded_query": {
            "id": degraded_q[0]["id"] if degraded_q else None,
            "question": degraded_q[0]["question"] if degraded_q else None,
            "relevant_docs": degraded_q[0]["relevant_docs"] if degraded_q else None,
            "mrr_before": degraded_q[1] if degraded_q else None,
            "mrr_after": degraded_q[2] if degraded_q else None,
        },
    })


def sweep_index() -> None:
    """Part D: Index and metadata.
    D1: ChromaRetriever vs DenseRetriever.
    D2: Scale latency profiling (NumPy vs HNSW).
    D3: Metadata filtering on trap questions (Q29-Q31).
    """
    corpus, questions = load_corpus(), load_questions()
    best_chunks = build_chunks(corpus, "markdown", size=400)

    # D1: ChromaRetriever vs DenseRetriever
    print("\n=== D1: Exact NumPy vs Chroma (HNSW) ===")
    r_dense = DenseRetriever(best_chunks, show_progress=False)
    m_dense = evaluate(r_dense, questions)

    r_chroma = ChromaRetriever(best_chunks, reset=True)
    m_chroma = evaluate(r_chroma, questions)

    d1_metrics = {
        "dense (exact NumPy)": m_dense,
        "chroma (HNSW ANN)": m_chroma,
    }
    cols = ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10", "latency_p95_ms")
    print(table(d1_metrics, cols=cols))
    recall_gap = m_dense["recall@5"] - m_chroma["recall@5"]
    print(f"  Recall gap (exact - ANN): {recall_gap:.4f}")

    # D2: Scale experiment at ~160, ~4k, ~40k chunks
    print("\n=== D2: Search Latency Scaling (NumPy vs HNSW) ===")
    dim = 3072
    scales = [len(best_chunks), 4000, 40000]
    timing_results = {}
    import chromadb
    client = chromadb.Client()

    for n in scales:
        mat = np.random.randn(n, dim).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        q = np.random.randn(dim).astype(np.float32)
        q /= np.linalg.norm(q)

        # Time NumPy exact
        times_np = []
        for _ in range(50):
            t0 = time.perf_counter()
            scores = mat @ q
            _ = np.argsort(-scores)[:8]
            times_np.append((time.perf_counter() - t0) * 1000)

        # Time Chroma HNSW
        col_scale = client.create_collection(f"scale_{n}_{int(time.time()*1000)%100000}", metadata={"hnsw:space": "cosine"})
        batch_add = 5000
        for b_start in range(0, n, batch_add):
            b_end = min(b_start + batch_add, n)
            col_scale.add(
                ids=[f"id_{i}" for i in range(b_start, b_end)],
                embeddings=[mat[i].tolist() for i in range(b_start, b_end)],
            )
        times_hnsw = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = col_scale.query(query_embeddings=[q.tolist()], n_results=8)
            times_hnsw.append((time.perf_counter() - t0) * 1000)

        p50_np = statistics.median(times_np)
        p50_hnsw = statistics.median(times_hnsw)
        timing_results[str(n)] = {"numpy_ms": p50_np, "hnsw_ms": p50_hnsw}
        print(f"  Scale {n:>5} chunks: Exact NumPy = {p50_np:6.3f} ms, HNSW = {p50_hnsw:6.3f} ms (Ratio: {p50_hnsw/p50_np:.2f}x)")

    # D3: Metadata filtering on trap questions (Q29, Q30, Q31)
    print("\n=== D3: Metadata Filtering (Archive Trap Q29-Q31) ===")
    tagged_chunks = []
    for c in best_chunks:
        meta = dict(c.meta)
        meta["status"] = "archived" if "ARCHIVED" in c.doc_id else "current"
        tagged_chunks.append(Chunk(c.text, c.doc_id, c.chunk_id, meta))

    r_chroma_tagged = ChromaRetriever(tagged_chunks, reset=True, collection="corpus_tagged")
    trap_ids = {"Q29", "Q30", "Q31"}
    trap_questions = [q for q in questions if q["id"] in trap_ids]

    # Evaluate before (unfiltered) on trap questions
    hits_before_h1 = []
    for q in trap_questions:
        hits = r_chroma_tagged.search(q["question"], k=1)
        retrieved_doc = hits[0].doc_id if hits else ""
        is_hit = 1.0 if retrieved_doc in q["relevant_docs"] else 0.0
        hits_before_h1.append(is_hit)
        print(f"  [Unfiltered] {q['id']}: retrieved '{retrieved_doc}', gold={q['relevant_docs']}, hit@1={is_hit}")

    # Evaluate after (filtered where={"status": "current"}) on trap questions
    hits_after_h1 = []
    for q in trap_questions:
        hits = r_chroma_tagged.search(q["question"], k=1, where={"status": "current"})
        retrieved_doc = hits[0].doc_id if hits else ""
        is_hit = 1.0 if retrieved_doc in q["relevant_docs"] else 0.0
        hits_after_h1.append(is_hit)
        print(f"  [Filtered]   {q['id']}: retrieved '{retrieved_doc}', gold={q['relevant_docs']}, hit@1={is_hit}")

    h1_before = statistics.fmean(hits_before_h1)
    h1_after = statistics.fmean(hits_after_h1)
    print(f"\nTrap Questions Q29-Q31 hit_rate@1: Before={h1_before:.3f} -> After={h1_after:.3f}")

    save_sweep_data("index", {
        "d1_chroma_vs_exact": {
            "dense": {k: v for k, v in m_dense.items() if not k.startswith("_")},
            "chroma": {k: v for k, v in m_chroma.items() if not k.startswith("_")},
        },
        "d2_scale_latencies": timing_results,
        "d3_trap_questions": {
            "hit_rate_at_1_before": h1_before,
            "hit_rate_at_1_after": h1_after,
        },
    })


SWEEPS = {
    "chunking": sweep_chunking,
    "retrieval": sweep_retrieval,
    "rerank": sweep_rerank,
    "index": sweep_index,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--sweep", choices=list(SWEEPS))
    args = ap.parse_args()
    if args.baseline or not args.sweep:
        sweep_baseline()
    if args.sweep:
        SWEEPS[args.sweep]()


if __name__ == "__main__":
    main()
