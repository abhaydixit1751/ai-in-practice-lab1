# Lab 7 Capstone Report — Ship It
**Course:** AI in Practice (Module 1: Applied GenAI & Production RAG)  
**Deliverables:** `labs/lab7/service.py`, `labs/lab7/ui.py`, `labs/lab7/dashboard.py`, `labs/lab7/gate.py`, `labs/lab7/thresholds.yml`, `labs/lab7/sweep_cache.py`, `.github/workflows/eval.yml`, `labs/lab7/REPORT.md`  
**Evaluation Benchmark:** 45-Question Golden Set (`data/eval/rag_golden.jsonl`)  
**Target Architecture:** FastAPI HTTP Service, Server-Sent Events Streaming, Dual-Layer Caching, Tracing & Ops Dashboard, CI Regression Gate  

---

## Executive Overview: The Module Capstone

Six labs developed individual system layers:
1. Pydantic contracts and structured output (Lab 1)
2. Model routing, evaluation grids, and cost accounting (Lab 2)
3. Lexical, dense, and hybrid retrieval sweeps with chunking strategies (Lab 3)
4. Grounded answer generation, citation contracts, and judge calibration (Lab 4)
5. Failure taxonomy diagnosis, Pareto analysis, and RAG v2 fixes (Lab 5)
6. Tool loops, schema validation boundaries, and red-team defenses (Lab 6)

**Lab 7 transforms these isolated components into a shippable production service.** It delivers a hardened HTTP API with an agreed Service Level Objective (p95 latency $\le 6,000$ ms, cost $\le \$0.01$/query), dual-layer caching, Server-Sent Events (SSE) streaming, local JSONL trace observability, operational alert conditions, and an automated continuous-integration regression gate running offline replay against committed artifacts.

---

## 1. What It Does (Non-Technical Executive Summary)

The **Aurora Policy Assistant** is an automated customer and staff advisor that answers questions about health insurance policies, claim filing procedures, waiting periods, room-rent limits, and exclusions. Unlike standard conversational AI that synthesizes answers from unverified general training data, this system references **only** official, active Aurora policy documents. 

For every factual statement made, the assistant generates an explicit, clickable citation linked directly to the specific policy document and section. If an inquiry addresses topics outside Aurora's documentation, or if the corpus lacks sufficient evidence to support an answer, the assistant deterministically declines to answer rather than guessing, hallucinating, or making unsupported assumptions.

---

## 2. System Architecture & Part-by-Part Implementation

The application implements the canonical seven-layer architecture (T1 §4):

```
Layer 7 · INTERFACE       FastAPI (/ask, /ask/stream), Streamlit UI (ui.py), Ops Dashboard (dashboard.py)
Layer 6 · ORCHESTRATION   ToolGuard budget enforcement, retry backoff with full jitter
Layer 5 · VALIDATION      Pydantic schemas, regex PII redaction, heuristic injection detection, citation check
Layer 4 · MODEL           LiteLLM client (gemini-3.5-flash-lite / gemini-3.7-flash), temperature=0.0
Layer 3 · CONTEXT         Markdown chunking (800 chars), DenseRetriever, document deduplication reranker
Layer 2 · DATA            Policy corpus (data/corpus/*.md), content-addressed SQLite cache (.aip_cache/)
Layer 1 · OBSERVABILITY   Structured JSONL tracing (.aip_traces/*.jsonl), global budget accounting
```

### Part A — The HTTP Service (`labs/lab7/service.py`)

#### A1. Contract & Response Integrity
The API contract is strictly typed via Pydantic:
- **Request (`AskRequest`):** `question` (3–1,000 chars), `top_k` (1–20, default 5), `mode` (`rag` or `tools`).
- **Response (`AskResponse`):** `answer` (str), `refused` (bool), `citations` (`list[Citation]`), `latency_ms` (float), `cost_usd` (float), `cached` (bool), `trace_id` (str).

*Architectural Rationale:* `cost_usd` and `trace_id` are delivered in the response body on purpose. The trace ID enables instant lookup in production spans without grepping timestamps, and the per-query cost makes financial expenditure visible directly to API consumers.

#### A2. Pipeline Lifecycle & Guardrail Integration
- **Startup Singleton (`pipeline()`):** To prevent re-embedding the 30-document corpus on every incoming HTTP call (which causes 40-second latency spikes), the retriever and RAG pipeline are constructed **once** at process startup and cached in memory.
- **Layered Guards on Request Path:**
  1. *PII Redaction (`redact_pii`):* Strips Aadhaar, PAN, phone numbers, and emails before vector embedding or prompt assembly.
  2. *Injection Defense (`detect_injection`):* Heuristic pattern detector checks for delimiter escapes, instruction overrides, and role switching. Suspicious inputs trigger safe refusal with `guard-refusal` trace ID.
  3. *Context Isolation (`delimit_untrusted`):* Encloses all retrieved documents in `<RETRIEVED_DOCUMENT>` XML boundaries, neutralizing indirect prompt injection.
  4. *Citation Validation (`enforce_citations`):* Verifies all `[n]` citation tags match valid indices in the retrieved document set ($1 \le n \le k$).
  5. *Tool Execution Guard (`ToolGuard`):* Active when `mode="tools"`, enforcing an allowlist (`search_policy`, `get_policy_details`, `compute_premium`), argument schema validation, and a strict 6-call ceiling.

#### A3. Status-Code Semantics & Graceful Degradation
| Scenario | HTTP Code | Implementation & Rationale |
|---|---|---|
| **Malformed Request** | **422** | Handled natively by Pydantic validation (e.g. empty queries or strings $< 3$ characters). |
| **Budget Exhaustion** | **429** | `BudgetExceeded` exception handler returns HTTP 429 to stop runaway caller loops. |
| **Model Outage / Rate Limit** | **503** | Upstream provider connection errors, timeouts, or 429 quota exhaustion are caught and translated into HTTP 503 with a **`Retry-After: 30`** header. Stack traces are never leaked. |
| **Internal Server Error** | **500** | Reserved solely for unexpected code-level defects. |

#### A4. User Interface (`labs/lab7/ui.py`)
- Built in Streamlit with expandable source citation drawers (`with st.expander(...)`).
- Allows users to inspect the underlying source text for every citation marker.
- Integrated Server-Sent Events streaming toggle.
- Added a thumbs-down feedback button that writes problematic cases to `data/review_queue.jsonl` for continuous evaluation set generation.

---

### Part B — Caching, Streaming, and Latency Budget

#### B1. Dual-Layer Caching Architecture
Two distinct caching layers with different risk profiles:

| Layer | Key Mechanism | Hit Rate | Latency | Risk Profile |
|---|---|:---:|:---:|---|
| **Exact Cache** | SHA-256 hash of normalized lowercase text | 15–25% | **< 10 ms** | Zero risk. Identical input guarantees identical grounded output. |
| **Semantic Cache** | Cosine similarity over normalized embedding vectors | +10–20% | **< 15 ms** | Risk of semantic collision: returning an answer to a different question. |

#### Empirical Semantic Cache Sweep (`labs/lab7/sweep_cache.py`)
To prevent silent semantic corruption, we systematically swept cosine thresholds against true paraphrases and near-miss collisions (plan substitutions and parameter differences):

```
Similarity         Label  Base Query                                Test Query                              
--------------------------------------------------------------------------------------------------------------
    0.9749    PARAPHRASE  How many days do I have to submit a re    What is the deadline for submitting a   
    0.9577    PARAPHRASE  What is the room rent limit on the Sil    How much room rent is allowed per day   
    0.9773    PARAPHRASE  Is maternity covered on the Bronze pla    Does the Bronze plan provide coverage   
    0.9421    PARAPHRASE  How long is the grace period for an an    What is the renewal grace period for a  
    0.8908    PARAPHRASE  What is the waiting period for pre-exi    How many months must I wait before pre  
    0.8782     COLLISION  What is the room rent limit on the Sil    What is the room rent limit on the Gol  
    0.8824     COLLISION  What is the room rent limit on the Sil    What is the room rent limit on the Pla  
    0.8479     COLLISION  Is maternity covered on the Bronze pla    Is maternity covered on the Gold plan?  
    0.8676     COLLISION  What is the waiting period for pre-exi    What is the waiting period for specifi  
    0.6530     COLLISION  What is the waiting period for pre-exi    What is the waiting period for joint r  
    0.8554     COLLISION  What is the ambulance cover on the Bro    What is the ambulance cover on the Pla  
```

```
Threshold Sweep Results:
  Threshold    Paraphrase Hit Rate    Collision / Wrong Hit Rate    Verdict
  ─────────────────────────────────────────────────────────────────────────────
    0.85             100.0%                     66.7%               UNSAFE (4 collisions)
    0.88             100.0%                     16.7%               UNSAFE (1 collision)
    0.90              80.0%                      0.0%               SAFE
    0.92              80.0%                      0.0%               SAFE
    0.94              80.0%                      0.0%               SAFE
    0.95              60.0%                      0.0%               SAFE
    0.96              40.0%                      0.0%               SAFE (Production)
```

> **Key Finding:** The breaking threshold where semantic caching begins answering the *wrong question* is **0.8824** (where Silver room rent limit collides with Platinum). Below 0.89, minor phrasing changes that alter the target insurance tier collide in vector space. We enforce a conservative production threshold of **0.96**.

#### B2 & B3. Streaming & The Citation Validation Problem
- Implemented `POST /ask/stream` using Server-Sent Events (`sse-starlette`).
- **The Core Dilemma:** Tokens stream to the client as they are generated, but citations cannot be validated until the answer text is complete.
- **Resolution & Defense (Option 3):**
  We stream prose tokens immediately to achieve a Time-to-First-Token (TTFT) of **840 ms** (well within the 1,500 ms target). Citation markers (`[1]`, `[2]`) are rendered in prose while the stream progresses. When generation terminates, the server executes `enforce_citations()`. A final `done` event delivers validated citation excerpts and the `citations_valid` boolean flag. In the UI, citation expanders render upon stream termination. If validation fails, the UI renders a warning banner advising that specific claims lack verified citations.

#### B4. Latency Budget Stage Breakdown
Measured from local spans on uncached requests:

```
Stage                  Latency (ms)    Share (%)    Notes
─────────────────────────────────────────────────────────────────────────────
1. Query Embedding           1.8 ms       0.1%      Gemini embedding API / local cache
2. Dense Retrieval           4.1 ms       0.2%      Cosine search across 164 chunks
3. Deduplication Rerank      0.1 ms      <0.1%      Document-level pool filtering
4. LLM Generation        1,843.7 ms      99.7%      Autoregressive token generation
5. Citation Validation       0.3 ms      <0.1%      Regex verification against hits
─────────────────────────────────────────────────────────────────────────────
Total End-to-End         1,850.0 ms     100.0%      SLO Target: <= 6,000 ms
```

> **Optimization Finding:** Generation dominates **99.7%** of uncached runtime. Optimizing vector search or chunking cannot move end-to-end latency by more than 5 milliseconds. The only meaningful latency optimizations are:
> 1. Restricting output token ceilings (capping at 200 tokens saves ~400 ms).
> 2. Streaming via SSE, which reduces user-perceived waiting time from 1,850 ms down to **840 ms**.

---

### Part C — Observability & Operations Dashboard (`labs/lab7/dashboard.py`)

#### C1 & C2. Tracing and `/metrics`
- Every request emits structured spans via `aip.tracing` recording `span_id`, `parent_id`, `run_id`, `name`, `duration_ms`, `status`, and `cost_usd`.
- Any slow query can be diagnosed immediately by filtering its trace ID across child spans (`retrieve`, `rerank`, `llm.call`).
- `GET /metrics` dynamically derives:
  - `total_queries`, `cache_hit_rate` ($30\%$), `cost_per_query_usd` ($\$0.00028$)
  - Latency percentiles: p50 ($1,240$ ms uncached, $4.2$ ms cached), p95 ($1,850$ ms uncached, $9.5$ ms cached), p99 ($2,410$ ms)
  - `error_rate_by_type` and `tool_call_counts`

#### C3 & C4. Dashboard & Operational Runbooks
The Streamlit dashboard (`labs/lab7/dashboard.py`) visualizes spans, cumulative spend curves, and stage latency tables. It integrates two active alert monitors:
1. **Refusal Rate Doubling Alert (Threshold: > 25%):**
   - *Failure Mode Detected:* Index degradation or missing corpus files. In RAG systems, a broken index returns zero matches and causes the model to decline politely without throwing an exception or HTTP 500. Watching refusal rate doubling detects silent data failures immediately.
   - *Runbook Action:* Query `/health` to verify index chunk count ($n=164$), inspect `data/corpus/` file integrity, and re-warm embeddings.
2. **p95 Latency SLO Breach Alert (Threshold: > 6,000 ms):**
   - *Failure Mode Detected:* Upstream provider degradation, network throttling, or generation runaway.
   - *Runbook Action:* Inspect provider status, review token length distributions in traces, and enforce output token caps.

---

### Part D — The Regression Gate (`labs/lab7/gate.py` & `thresholds.yml`)

#### D1 & D2. Evaluation Gate & Offline CI Replay
The evaluation regression gate evaluates all 45 questions in the golden dataset. It runs under `AIP_OFFLINE=1` against `.aip_cache/calls.sqlite3`, executing in **under 1.0 second** at **$0.00 cost** without requiring an external API key.

Thresholds in `thresholds.yml` were tuned to provide roughly 1 standard error of headroom below achieved performance:

```yaml
correctness:        {min: 0.75}
faithfulness:       {min: 0.90}
citation_validity:  {min: 0.98}
refusal_recall:     {min: 0.80}
refusal_precision:  {min: 0.60}
hit_rate_at_5:      {min: 0.85}
cost_per_query_usd: {max: 0.010}
p95_latency_ms:     {max: 6000}
```

#### D3. Proving the Gate Works: The Red Build
> *"A gate you have not seen fail is a gate you do not have."*

We verified gate functionality under both normal and broken conditions:

1. **Normal Run (`python labs/lab7/gate.py`):**
   ```text
   metric                   value            gate  status
   ----------------------------------------------------------
   correctness             0.8875         >= 0.75  ok
   faithfulness            0.9778          >= 0.9  ok
   citation_validity       1.0000         >= 0.98  ok
   refusal_recall          1.0000          >= 0.8  ok
   refusal_precision       0.6250          >= 0.6  ok
   hit_rate_at_5           1.0000         >= 0.85  ok
   cost_per_query_usd      0.0003         <= 0.01  ok
   p95_latency_ms          5.4000         <= 6000  ok

   GATE PASSED (Exit Code: 0)
   ```

2. **Deliberate Break Run (`python labs/lab7/gate.py --break`):**
   Simulates index corruption and ungrounded generation. The gate catches six simultaneous violations and exits with non-zero status code 1:
   ```text
   metric                   value            gate  status
   ----------------------------------------------------------
   correctness             0.0000         >= 0.75  FAIL
   faithfulness            0.0000          >= 0.9  FAIL
   citation_validity       0.0000         >= 0.98  FAIL
   refusal_recall          0.0000          >= 0.8  FAIL
   refusal_precision       0.0000          >= 0.6  FAIL
   hit_rate_at_5           0.0000         >= 0.85  FAIL
   cost_per_query_usd      0.0003         <= 0.01  ok
   p95_latency_ms         50.0000         <= 6000  ok

   GATE FAILED:
     correctness: 0.0 violates >= 0.75
     faithfulness: 0.0 violates >= 0.9
     citation_validity: 0.0 violates >= 0.98
     refusal_recall: 0.0 violates >= 0.8
     refusal_precision: 0.0 violates >= 0.6
     hit_rate_at_5: 0.0 violates >= 0.85
   (Exit Code: 1)
   ```

3. **CI Integration:** Created [`.github/workflows/eval.yml`](file:///c:/Users/Acer/Desktop/AI-in-Practice-Lab\aip-lab1\.github\workflows\eval.yml) to run this gate on all GitHub pushes and pull requests.

---

## 3. How Well It Works (Full Metric Scorecard)

Evaluated across the 45-question golden benchmark:

| Metric | Target / SLO | Baseline (Lab 4) | Production (Lab 7) | Status |
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

---

## 4. Where It Fails (Failure Mode Analysis)

Five out of 45 benchmark questions did not achieve full 2/2 correctness scores. Categorized into the T4 §5 diagnostic tree:

```
Remaining Failure Taxonomy (Total: 5)
  Mode 4: Multi-hop Ranking Loss    ███████████████ 3 cases (60%) [Q11, Q20, Q32]
  Mode 6: Generation Incompleteness ██████████      2 cases (40%) [Q10, Q29]
```

### Detailed Breakdown:
1. **Mode 4: Cross-Document Ranking Loss (Q11, Q20, Q32 — 60% of failures):**
   - *Example (Q20):* Requires comparing terms across `plans-overview`, `critical-illness-riders`, and `exclusions`. Single-pass cosine retrieval scores paragraphs from the first two documents highly, but ranks the necessary exclusion paragraph 7th, causing it to fall outside `final_k=5`.
   - *Remediation:* Multi-query paraphrase expansion or reciprocal rank fusion across sub-queries.
2. **Mode 6: Generation Incompleteness (Q10, Q29 — 40% of failures):**
   - *Example (Q29):* In pre-existing disease continuity rules, the generator stated the core 36-month waiting period accurately with citations, but omitted a minor portability transfer exception detailed at the end of the clause.
   - *Remediation:* Adjusting prompt length constraints to permit full clause enumeration when exceptions are present.

---

## 5. What It Costs (Financial Model)

- **Input Token Average:** 1,650 tokens / uncached query
- **Output Token Average:** 115 tokens / uncached query
- **Blended Cache Hit Rate:** 30% (20% exact hash + 10% semantic cache)

### Cost Scaling Table
| Volume Tier | Uncached Spend | With 30% Cache | Annualized Total |
|:---|:---:|:---:|:---:|
| **Single Query** | $0.000280 | $0.000196 | — |
| **1,000 Queries** | $0.280 | $0.196 | — |
| **Daily (10,000 queries/day)** | $2.80 | $1.96 | **$715.40 / year** |
| **High Traffic (50,000 queries/day)** | $14.00 | $9.80 | **$3,577.00 / year** |

*Conclusion:* The service operates at **$1.96 per day** for 10,000 daily queries. It sits comfortably below the budget ceiling of $0.01/query.

---

## 6. What It Is NOT Safe For (Boundaries of Safe Use)

Stating what a system cannot safely do is essential for responsible deployment:

1. **Automated Binding Coverage Authorizations:**
   - *Prohibition:* The assistant must not be granted authority to issue legally binding pre-authorizations or claim approval guarantees.
   - *Reason:* While citation accuracy is 100%, insurance claim adjudication requires verifying hospital accreditation tiers, doctor notes, and historical policy lapse dates that reside in external ERP databases outside the static policy corpus.
2. **Emergency Care & Clinical Triage:**
   - *Prohibition:* Must not be placed in emergency department intake paths.
   - *Reason:* Network latencies (p95 = 1.85 s) and upstream provider rate limit risks are incompatible with life-critical clinical response requirements.
3. **Unsupervised Financial Transactions:**
   - *Prohibition:* Automated refund issuance without human sign-off.
   - *Reason:* Although capped at ₹50,000 in `RefundArgs`, any automated financial transaction tool exposed to natural language is susceptible to sophisticated social-engineering attacks. Human confirmation (`ToolGuard.requires_confirmation`) is mandatory.

---

## 7. What We Would Do Next (Ranked Roadmap)

1. **Multi-Query Decomposition for Cross-Document Retrieval (Expected Value: +8% Correctness):**
   - *Plan:* Decompose complex multi-hop queries into sub-questions before retrieval. This addresses all three Mode 4 ranking failures (Q11, Q20, Q32).
2. **Entity-Scoped Semantic Cache Keys (Expected Value: +15% Cache Hit Rate):**
   - *Plan:* Prefix semantic cache keys with extracted entities: `(plan_tier, question_embedding)`. This prevents cross-plan collisions, allowing the cosine threshold to be safely relaxed from 0.96 down to 0.90.
3. **Continuous Golden Set Expansion via Review Queue (Expected Value: +100 Golden Cases/Month):**
   - *Plan:* Ingest cases flagged by the UI thumbs-down button directly into an evaluation triage pipeline, automating test set expansion from live operational traffic.

---

## 8. Demo Day Checklist & Script

| Item | Focus | Prepared Action & Evidence |
|:---:|---|---|
| **1** | **Question answered well (30s)** | Query: *"How many days do I have to submit a reimbursement claim after discharge?"*<br>Result: Answers *"30 days from discharge"* with valid citations `[2][5]`. |
| **2** | **Question refused (30s)** | Query: *"Does the policy cover experimental robotic surgery performed abroad?"*<br>Result: Emits exact refusal string `"I don't have enough information in the provided sources to answer that."` Citations valid, hallucination avoided. |
| **3** | **Show `/metrics` (1m)** | Display live metrics endpoint: Cache hit rate $30\%$, uncached p95 $1,850$ ms, cached p95 $9.5$ ms, average cost $\$0.00028$. |
| **4** | **Regression Gate Failure (1m)** | Run `python labs/lab7/gate.py --break`. Show terminal output failing 6 metric thresholds and exiting with code 1. |
| **5** | **Worst Remaining Failure (1m)** | Present Q20 (multi-hop ranking loss across 3 documents). Explain root cause (Mode 4) and remediation via multi-query decomposition. |
| **6** | **Questions (1m)** | Address latency budget breakdown (99.7% generation) and the 0.8824 semantic cache breaking threshold. |

---

### Verification Record
- Service API tests (`POST /ask`, `POST /ask/stream`, `GET /health`, `GET /metrics`): **Passed**
- Error handling tests (422, 429, 503 + `Retry-After: 30`): **Passed**
- Regression gate normal execution (`python labs/lab7/gate.py`): **Passed (Exit Code 0)**
- Regression gate deliberate break execution (`python labs/lab7/gate.py --break`): **Failed as intended (Exit Code 1)**
- Ruff style and syntax checks: **Clean (All checks passed)**
- Existing suite unit tests: **35 passed in 0.34s**
