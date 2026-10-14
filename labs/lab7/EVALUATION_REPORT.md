# Aurora Health Policy Assistant — Capstone Evaluation Report
**System Version:** 1.0 (Production Release)  
**Authors / Pairs:** AI in Practice Engineering Team  
**Evaluation Benchmark:** 45-Question Aurora Golden Set (`data/eval/rag_golden.jsonl`)  
**Deployment Target:** FastAPI + Server-Sent Events (`POST /ask`, `POST /ask/stream`), Streamlit Ops Dashboard  

---

## 1. What It Does (Executive Summary)

The Aurora Policy Assistant is an automated customer and staff advisor that answers questions about health insurance plans, claim procedures, waiting periods, and benefit limits. Unlike traditional chatbots that generate answers from unverified internal memory, this system references only official, active Aurora policy documents. For every factual claim made in an answer, the assistant provides a verifiable citation linking directly to the specific policy clause. When a question addresses topics not covered in Aurora's documentation, or when necessary policy details are missing, the assistant explicitly declines to answer rather than guessing or fabricating coverage rules.

---

## 2. How Well It Works (Evaluation Metrics)

The system was evaluated against the 45-question golden benchmark (`data/eval/rag_golden.jsonl`), spanning single-hop retrieval, complex multi-hop comparisons, document aggregations, and intentionally unanswerable inquiries.

### Headline Metric Summary

| Metric | Target / SLO | Baseline (v1) | Production (v2) | Status |
|:---|:---:|:---:|:---:|:---:|
| **Citation Validity** | $\ge 0.980$ | 0.978 | **1.000** (45/45) | **Passed** |
| **Faithfulness** | $\ge 0.900$ | 0.933 | **0.978** (44/45) | **Passed** |
| **Answer Correctness** | $\ge 0.750$ | 0.825 | **0.888** (35.5 / 40) | **Passed** |
| **Refusal Recall** | $\ge 0.800$ | 1.000 (5/5) | **1.000** (5/5) | **Passed** |
| **Refusal Precision** | $\ge 0.600$ | 0.556 (5/9) | **0.625** (5/8) | **Passed** |
| **Retrieval Hit Rate @ 5** | $\ge 0.850$ | 0.887 | **1.000** (40/40) | **Passed** |
| **Cost per Query** | $\le \$0.0100$ | $\$0.00030$ | **\$0.00028** | **Passed (35× below ceiling)** |
| **p95 Latency (Cached)** | $\le 800$ ms | N/A | **9.5 ms** | **Passed (84× faster)** |
| **p95 Latency (Uncached)** | $\le 6,000$ ms | 4,259 ms | **1,850 ms** | **Passed (3.2× faster)** |

### Caching Architecture & Empirical Threshold Sweep (Part B1)
The service implements a dual-layer caching strategy:
1. **Exact Hash Cache:** Content-addressed SHA-256 hash of normalized queries. Captures 15–25% of identical requests, returning verified answers in under 10 ms at $0.00 cost.
2. **Semantic Cache:** Cosine similarity over query vector embeddings (`gemini-embedding-001`). Captures phrasing variations.

#### Measured Semantic Cache Collision Sweep
| Cosine Threshold | Paraphrase Hit Rate | Collision / Wrong Hit Rate | Safety Verdict |
|:---:|:---:|:---:|:---:|
| **0.85** | 100.0% | 66.7% | ❌ **Unsafe** (confuses Silver/Gold/Platinum room rents) |
| **0.88** | 100.0% | 16.7% | ❌ **Unsafe** (wrong hit on Silver vs Platinum at 0.8824) |
| **0.90** | 80.0% | 0.0% | ✅ **Safe** (Zero false hits observed) |
| **0.92** | 80.0% | 0.0% | ✅ **Safe** (Recommended conservative baseline) |
| **0.95** | 60.0% | 0.0% | ✅ **Safe** (Reduced hit rate) |
| **0.96** | 40.0% | 0.0% | ✅ **Safe** (Production setting) |

> **Critical Empirical Finding:** The breaking threshold where semantic caching begins answering the *wrong question* is **0.8824**. Below 0.89, queries differing by a single critical parameter (e.g., plan tier: "Silver" vs "Gold") collide in cosine space. We enforce a production threshold of **0.96**, backed by exact plan-name scoping.

---

## 3. Where It Fails (Failure Taxonomy & Backlog)

Across the 45-question golden set, exactly 5 queries failed to achieve full 2/2 correctness scores. All remaining failures belong to two specific modes in the T4 §5 taxonomy:

```
Remaining Failure Distribution (Total Failures: 5)
  Mode 4: Multi-hop Ranking Loss    ███████████████ 3 cases (60%) [Q11, Q20, Q32]
  Mode 6: Generation Incompleteness ██████████      2 cases (40%) [Q10, Q29]
```

### Specific Failure Analysis:
1. **Mode 4 (Ranking Loss across Dispersed Documents — Q11, Q20, Q32):**
   - *Symptom:* Questions that require synthesizing clauses from three disparate policy files (e.g., Q20 requires definitions from `plans-overview`, `critical-illness-riders`, and `exclusions`).
   - *Root Cause:* Although individual documents appear in the top-15 candidate pool, dense similarity alone does not score cross-document bridge queries high enough to keep all three in the top-5 context window without displacing one another.
2. **Mode 6 (Generation Incompleteness on Multi-condition Clauses — Q10, Q29):**
   - *Symptom:* On Q29 (pre-existing disease waiting period exceptions), the model provided the general 36-month waiting rule and cited it properly, but omitted the 12-month accelerated portability waiver clause present in the second half of the excerpt.
   - *Root Cause:* Generation length penalty prompt constraint (keeping prose concise within 2–3 sentences) inadvertently penalized exhaustive multi-clause enumeration.

---

## 4. What It Costs (Financial Accounting)

Using token telemetry from `aip.cost` and live API rate cards:
- **Input Tokens (Prompt + Context):** ~1,650 tokens per uncached request
- **Output Tokens (Grounded Prose):** ~115 tokens per uncached request
- **Blended Cache Hit Rate:** 30% (20% exact cache + 10% semantic cache)

### Financial Model
| Volume Tier | Uncached Cost | With 30% Cache | Projected Cost |
|:---|:---:|:---:|:---:|
| **Per Single Query** | $0.000280 | $0.000196 | **<$0.0002** |
| **Per 1,000 Queries** | $0.280 | $0.196 | **$0.20** |
| **Daily (10,000 queries/day)** | $2.80 | $1.96 | **$1.96 / day** |
| **Annual (3.65M queries/year)** | $1,022.00 | $715.40 | **$715.40 / year** |

*Conclusion:* The entire system operates at less than $720 per year under sustained enterprise traffic (10,000 queries daily), leaving 98% headroom below the $0.01/query maximum budget ceiling.

---

## 5. How Fast It Is (Latency Budget Breakdown)

Latency distributions exhibit extreme skew between cold uncached requests and warm cache hits.

| Percentile | Cached End-to-End | Uncached End-to-End | Production SLO |
|:---|:---:|:---:|:---:|
| **p50 (Median)** | **4.2 ms** | **1,240 ms** | $\le 3,000$ ms |
| **p95** | **9.5 ms** | **1,850 ms** | $\le 6,000$ ms |
| **p99 (Tail)** | **18.0 ms** | **2,410 ms** | $\le 8,000$ ms |
| **Streaming TTFT** | **1.2 ms** | **840 ms** | $\le 1,500$ ms |

### Uncached p95 Stage Breakdown
```
Stage                  Latency (ms)    Share (%)
─────────────────────────────────────────────────
1. Query Embedding           1.8 ms       0.1%
2. Dense Retrieval           4.1 ms       0.2%
3. Deduplication Rerank      0.1 ms      <0.1%
4. LLM Generation        1,843.7 ms      99.7%  ██████████████████████
5. Citation Validation       0.3 ms      <0.1%
─────────────────────────────────────────────────
Total End-to-End         1,850.0 ms     100.0%
```

> **Optimization Priority:** Generation accounts for **99.7%** of uncached response latency. Optimizing retrieval or chunking cannot move p95 by more than 6 milliseconds. The primary levers for reducing p95 latency are:
> 1. Speculative early decoding / max token capping (restricting output to 200 tokens saves ~400 ms).
> 2. Streaming with Server-Sent Events (`POST /ask/stream`), which drops perceived TTFT from 1,850 ms down to **840 ms**.

---

## 6. What It Is NOT Safe For (Boundaries of Safe Use)

A responsible AI engineering document must articulate where automation ceases to be trustworthy. This assistant operates under explicit boundaries:

1. **Binding Pre-Authorizations & Coverage Determinations:**
   - *Prohibition:* The assistant must **never** be used as an automated claims adjudicator or to issue binding coverage guarantees to patients or hospitals.
   - *Reason:* While citation validity is 100%, health policies contain overlapping rider schedules, active medical exclusions, and pre-existing declaration clauses that depend on audited medical records outside this corpus.
2. **Emergency Care Protocol Decisions:**
   - *Prohibition:* Must not be deployed in emergency admission or triage workflows without human staff intermediation.
   - *Reason:* Uncached p95 latency (~1.8 seconds) and potential upstream provider rate limits make synchronous HTTP dependencies unacceptable in clinical critical-path workflows.
3. **Out-of-Distribution or Non-Policy Inquiries:**
   - *Prohibition:* Questions regarding medical diagnosis, treatment alternatives, or competitor insurance plan comparisons.
   - *Reason:* The system strictly enforces policy document grounding; non-policy medical reasoning triggers either safe refusal or incomplete guidance.

---

## 7. What We Would Do Next (Ranked Roadmap)

1. **Multi-Hop Query Decomposition with Sub-Query Routing (Expected Value: +8% Correctness, High Impact):**
   - *Plan:* For complex comparative queries, use an orchestrator to decompose the prompt into discrete single-hop lookups before reciprocal rank fusion. This resolves remaining Mode 4 failures (Q11, Q20, Q32).
2. **Entity-Scoped Semantic Cache Keys (Expected Value: +15% Hit Rate, Medium Impact):**
   - *Plan:* Key semantic cache entries on `(plan_tier, question_embedding)`. This eliminates false-positive collisions between plans, allowing the cosine threshold to be safely reduced from 0.96 to 0.90 without risk.
3. **Active Client Feedback Review Queue Ingestion (Expected Value: +100 Golden Cases/Month, High Compounding Value):**
   - *Plan:* Wire the UI thumbs-down feedback directly into an automated regression suite generation pipeline, transforming real user edge cases into test cases within 24 hours.
