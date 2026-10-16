# Lab 3 Report — Semantic Search That Actually Works

**Course:** AI in Practice I · Module 1 (Reliable Systems)  
**Lab:** Lab 3 (Retrieval Engineering)  
**System Evaluated:** Aurora Health Support Semantic Policy Search Layer (Foundation for Labs 4–7)  
**Corpus:** 30 documents (16 Aurora Health policies + 14 lexical distractors) in `data/corpus/`  
**Evaluation Set:** 45 golden questions in `data/eval/rag_golden.jsonl` ($n=42$ evaluated, 3 unanswerable without gold documents excluded)  
**Artifacts Generated:** [`labs/lab3/search.py`](file:///c:/Users/Acer/Desktop/AI-in-Practice-Lab/aip-lab1/labs/lab3/search.py), [`reports/lab3_sweeps.json`](file:///c:/Users/Acer/Desktop/AI-in-Practice-Lab/aip-lab1/reports/lab3_sweeps.json)

---

## 1. Lab Overview & Architecture: What We Have to Do

### 1.1 The Operational Problem
Aurora Health customer support agents currently search internal policy manuals using PDF text search (`Ctrl-F`). This approach systematically fails in enterprise environments due to vocabulary mismatch: agents search for colloquial phrasing (*"how long do I have to file"*) whereas formal policy contracts use legalistic phrasing (*"submitted within 30 days of discharge"*). `Ctrl-F` returns zero results, forcing manual scanning or incorrect customer guidance.

The objective of Lab 3 is to engineer, evaluate, and justify an end-to-end semantic retrieval layer that serves as the factual foundation for the RAG generation, refusal, and agentic workflows in Labs 4 through 7.

### 1.2 Corpus Composition & Lexical Distractors
The retrieval corpus consists of 30 markdown documents:
* **16 Core Policy Documents:** Authentic Aurora Health rules covering maternity benefits, claims processes, exclusions, room rent caps, pre-existing conditions, outpatient wellness, and specific plan tiers (Bronze, Silver, Gold, Platinum).
* **14 Lexical Distractors:** Non-applicable or obsolete policies (motor accident claim timelines, travel insurance exclusions, corporate group health comparisons, life insurance grace periods, senior citizen plans, mobile app guides, and archived policy versions).

Distractors are deliberately engineered to share high lexical overlap with genuine health inquiries. In a toy corpus without distractors, trivial keyword search achieves artificially high retrieval scores; distractors test whether the retrieval stack can distinguish genuine semantic relevance from lexical coincidence.

### 1.3 Evaluation Protocol: Why $n=42$, Not 45
The evaluation set contains 45 questions with document-level relevance judgements (`rag_golden.jsonl`). As required by the laboratory specification:
* **Questions Q36, Q38, and Q39 have no relevant documents in the corpus** (e.g., asking about cosmetic surgery coverage under basic plans or unlisted travel benefits).
* Computing rank-aware metrics (`recall@k`, `precision@k`, `nDCG@k`) against an empty gold set is mathematically undefined ($0/0$).
* These three questions are explicitly excluded from retrieval benchmarking, yielding an active evaluation size of **$n=42$**. (All five unanswerable questions, Q36–Q40, are evaluated in Lab 4 to measure hallucination refusal precision and recall).

### 1.4 Search Space & The Greedy Sweep Methodology
The combinatorial search space spans 5 distinct dimensions:
1. **Chunking Strategy:** Fixed, Sliding, Recursive, Markdown-aware (4 options)
2. **Chunk Size:** 400, 800, 1600 characters (3 options)
3. **Retrieval Mechanism:** Dense (cosine), BM25 (lexical), Hybrid (Reciprocal Rank Fusion) (3 options)
4. **Reranking Strategy:** None, Cross-Encoder (`ms-marco-MiniLM-L-6-v2`), LLM Reranker (`gemini-3.5-flash-lite`) (3 options)
5. **Index Type:** Exact NumPy matrix vs. ChromaDB HNSW (2 options)

An exhaustive evaluation grid ($4 \times 3 \times 3 \times 3 \times 2 = 216$ cells) is computationally impractical within a 3-hour laboratory window. We therefore execute a disciplined **greedy sweep**:
$$\text{Baseline} \longrightarrow \text{Chunking Sweep (Part A)} \longrightarrow \text{Retrieval Sweep (Part B)} \longrightarrow \text{Reranking Sweep (Part C)} \longrightarrow \text{Indexing \& Metadata (Part D)}$$
At each stage, the highest-performing configuration is frozen before sweeping the next axis.

> **Greedy Sweep Assumption & Risk:** Greedy optimization assumes that architectural dimensions are strictly **separable**—i.e., that the optimal chunker for dense retrieval is also optimal for BM25 or reranking. In practice, this assumption can fail when components interact non-linearly (for example, small chunk sizes benefit dense retrieval by reducing semantic dilution, but degrade BM25 term frequency statistics).

### 1.5 The Metric Satiation Trap
A central methodological lesson of this lab is metric headroom:
* On this corpus, **`hit_rate@5` is saturated at 0.93–0.98 across virtually all configurations**. Comparing retrievers on `hit_rate@5` falsely suggests "no meaningful difference" (e.g., Dense 0.976 vs. BM25 0.929).
* Using non-saturated metrics that penalize poor rank positions—specifically **Mean Reciprocal Rank (MRR)** and **nDCG@10**—reveals dramatic operational divergence (e.g., Dense MRR 0.8800 vs. BM25 MRR 0.6698).

---

## 2. Part 0: Baseline Evaluation

The provided baseline utilizes sliding-window chunking (800 characters, 150 overlap) with an exact NumPy `DenseRetriever` (`gemini-embedding-001`, 3072 dimensions):

* **Corpus Splitting:** 30 documents $\longrightarrow$ 91 chunks (mean 707 characters/chunk).
* **Baseline Scores ($n=42$):**

| Configuration | hit_rate@1 | hit_rate@5 | recall@5 | MRR | nDCG@10 | Latency p95 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (`sliding-800`, dense)** | **0.7857** | **0.9286** | **0.8452** | **0.8451** | **0.8053** | **2.09 ms** |

---

## 3. Part A — Chunking Strategy & Dilution Sweep

Chunking is the single highest-leverage decision in retrieval architecture. A document split across a critical clause cannot be retrieved by downstream models, regardless of embedding or reranker sophistication.

### 3.1 A1: Strategy Comparison (Fixed at 800 Characters)
All four chunking strategies were evaluated under identical conditions using exact dense retrieval:

| Strategy | Chunks | Build Time | hit_rate@1 | hit_rate@5 | recall@5 | MRR | nDCG@10 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `fixed-800` | 83 | 11.40 s | 0.7381 | 0.9524 | 0.8373 | 0.8387 | 0.7952 |
| `sliding-800` (150 ovlp) | 91 | 0.08 s | 0.7857 | 0.9286 | 0.8452 | 0.8451 | 0.8053 |
| `recursive-800` (100 ovlp) | 98 | 99.87 s | 0.7619 | 0.9524 | 0.8750 | 0.8611 | 0.8251 |
| **`markdown-800`** | **164** | **0.16 s** | **0.7619** | **0.9762** | **0.8988** | **0.8720** | **0.8458** |

*Findings:* `markdown-800` outperformed all competitors, improving nDCG@10 by +0.0506 over fixed splitting and +0.0405 over baseline sliding windows, while maintaining near-instant build times (0.16s).

### 3.2 A2: Chunk Size Sweep & The Dilution Curve
Holding the winning strategy (`markdown`) constant, chunk size was swept across $\{400, 800, 1600\}$ characters:

| Size | Chunks | hit_rate@1 | hit_rate@5 | recall@5 | MRR | nDCG@10 | Latency p95 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **400 chars** | **235** | **0.7857** | **0.9762** | **0.9028** | **0.8800** | **0.8527** | **2.41 ms** |
| 800 chars | 164 | 0.7619 | 0.9762 | 0.8988 | 0.8720 | 0.8458 | 2.16 ms |
| 1600 chars | 150 | 0.7143 | 0.9524 | 0.8750 | 0.8262 | 0.8075 | 3.49 ms |

```
nDCG@10
  0.860 ┼───────● (400 chars: 0.8527)
  0.840 ┼               ● (800 chars: 0.8458)
  0.820 ┼
  0.800 ┼                               ● (1600 chars: 0.8075)
        └───────┬───────┬───────┬───────┬───────▶ Chunk Size
               400     800    1200    1600
```

#### Theoretical Explanation: The Dilution Mechanism (T4 §2.2)
The relationship between chunk size and retrieval quality is non-monotonic:
* **The Dilution Problem at 1600 characters:** An embedding vector represents a single point in high-dimensional semantic space ($d=3072$). When a chunk expands to 1600 characters, it encompasses multiple heterogeneous policy clauses (e.g., waiting periods, disease exceptions, co-pay tiers). The resulting vector represents the mathematical centroid (average) of these divergent topics, drifting away from specific queries.
* **The Boundary Problem below 400 characters:** Excessively small chunks fragment compound legal rules across chunk boundaries, causing answers to lack necessary qualifying context.
* **The Knee at 400 characters:** For Aurora Health policies, 400 characters preserves single, complete operational clauses while maximizing semantic vector specificity.

### 3.3 A3: Heading-Path Prefix Ablation
`markdown_chunks` prepends the document hierarchical heading breadcrumb (e.g., `[Exclusions > Specific Treatments]\n`) to each chunk. We ablated this feature by stripping the bracketed prefix:

| Configuration | hit_rate@1 | hit_rate@5 | recall@5 | MRR | nDCG@10 |
|:---|:---:|:---:|:---:|:---:|:---:|
| `markdown-400` (with prefix) | **0.7857** | 0.9762 | 0.9028 | **0.8800** | **0.8527** |
| `markdown-400` (prefix stripped) | 0.6905 | **1.0000** | **0.9107** | 0.8131 | 0.8204 |
| **Delta ($\Delta$)** | **+0.0952 (+9.5%)** | **-0.0238 (-2.4%)** | **-0.0079** | **+0.0669** | **+0.0323** |

> **Diagnostic Finding:** Prepending the heading path significantly boosts **ranking precision** (`hit_rate@1` surges by +9.5 percentage points; MRR increases by +0.0669), while slightly reducing `hit_rate@5` (-2.4%). Context-free chunks often lack topical signposts; the heading prefix acts as an anchor that pulls top-relevant documents into the rank-1 position, converting broad recall into sharp top-1 accuracy.

### 3.4 A4: Failure Mode 2 Case Study (Chunk Boundary Fracture)
* **Question Q08:** *"Can I claim for IVF treatment?"*
* **Ground Truth Document:** `exclusions`
* **Fixed-800 Result (Failure):** Retrieved `['hospital-cash-benefit', 'maternity-benefits', 'claims-timelines-2024-ARCHIVED', 'claims-timelines', 'health-faq-general']`. Gold document completely absent from top 5.
* **Markdown-400 Result (Success):** Retrieved `['exclusions', 'maternity-benefits', 'health-faq-general', ...]` (Gold doc ranked #1).
* **Root Cause Analysis:** In `fixed-800`, the specific exclusion clause for fertility treatments was bisected across a fixed 800-character boundary. Without the section header, the isolated text snippet lost its semantic link to policy exclusions, allowing distractor chunks from `maternity-benefits` to outscore it. In `markdown-400`, the clause remained intact under `[Exclusions > Specific Treatments]`.

---

## 4. Part B — Dense vs. BM25 vs. Hybrid Retrieval

With chunking fixed to `markdown-400`, we evaluated the three primary retrieval mechanisms.

### 4.1 B1: Core Retriever Comparison

| Retriever | hit_rate@1 | hit_rate@5 | recall@5 | MRR | nDCG@10 | Latency p95 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dense (`markdown-400`)** | **0.7857** | **0.9762** | **0.9028** | **0.8800** | **0.8527** | **2.78 ms** |
| BM25 (`markdown-400`) | 0.4762 | 0.9286 | 0.7956 | 0.6698 | 0.6978 | **1.71 ms** |
| Hybrid RRF ($k=60$, 1:1) | 0.6667 | **0.9762** | 0.8631 | 0.7976 | 0.7949 | 4.91 ms |

### 4.2 B2: Question-Kind Breakdown (MRR)
Breaking down performance across query categories exposes the limitations of aggregate reporting:

| Question Kind | $n$ | Dense MRR | BM25 MRR | Hybrid MRR | Winning Architecture |
|:---|:---:|:---:|:---:|:---:|:---|
| `single_hop` | 18 | 0.9074 | 0.8519 | **0.9444** | Hybrid (+0.037 over Dense) |
| `multi_hop` | 10 | **1.0000** | 0.6500 | 0.8167 | **Dense (+0.183 over Hybrid)** |
| `aggregation` | 4 | **0.8750** | 0.3750 | 0.5833 | **Dense (+0.292 over Hybrid)** |
| `paraphrase` | 5 | **0.8000** | 0.4867 | 0.6500 | **Dense (+0.150 over Hybrid)** |
| `trap_archived` | 3 | **0.8333** | 0.5111 | 0.6667 | **Dense (+0.167 over Hybrid)** |
| `unanswerable` | 2 | 0.3125 | **0.4167** | 0.3750 | BM25 |

#### Mechanism Analysis: Q44 (Exact Identifier) vs. Q41 (Paraphrase)

| Question | Query Text / Target | Dense MRR | BM25 MRR | Hybrid MRR | Mechanism |
|:---|:---|:---:|:---:|:---:|:---|
| **Q44** | `AUR-HI-SIL-2026` (Exact policy identifier) | 0.5000 | **1.0000** | **1.0000** | **Fusion Rescues:** Dense representations struggle with unseen alphanumeric codes. BM25 matches exact string tokens via IDF. RRF successfully elevates the BM25 candidate to rank 1. |
| **Q41** | *"if I skip paying on time, how long before I lose everything"* (Paraphrase of "grace period") | **1.0000** | 0.0000 | 0.2500 | **Fusion Damages:** Query shares zero lexical overlap with policy prose. Dense matches semantic concept perfectly (Rank 1). BM25 fails completely. Fusing zero-scoring BM25 degrades rank from 1 to 4. |

### 4.3 B3 & B4: RRF Sensitivity & Weight Tuning
* **$k$-Parameter Sensitivity ($k \in \{10, 30, 60, 100\}$):**
  * $k=10$: nDCG@10 = 0.8156, MRR = 0.8115
  * $k=30$: nDCG@10 = 0.7949, MRR = 0.7976
  * $k=60$: nDCG@10 = 0.7949, MRR = 0.7976
  * $k=100$: nDCG@10 = 0.7901, MRR = 0.7976
  * *Verdict:* Performance variance across the $k$ grid is minimal ($\Delta \le 0.02$). This insensitivity confirms that RRF is parameter-stable.
* **Unequal Fusion Weights:**
  * Weight 1:1 $\longrightarrow$ nDCG@10 = 0.7949
  * Weight 2:1 (favoring Dense) $\longrightarrow$ nDCG@10 = 0.8068
  * Weight 3:1 (favoring Dense) $\longrightarrow$ nDCG@10 = 0.8136
  * *Verdict:* While tilting weights toward Dense mitigates corruption, no hybrid weighting matches pure Dense (0.8527). At $n=42$, fine-tuning weights fits sample noise.

### 4.4 B5: The Prominent Negative Finding — Why Hybrid Loses on This Corpus
In academic literature and industry benchmarks (e.g., T4 §4.3), hybrid search is widely promoted as "the single strongest improvement in RAG". **On this corpus, Hybrid is measurably inferior to pure Dense retrieval (nDCG@10: 0.7949 vs. 0.8527; MRR: 0.7976 vs. 0.8800).**

The underlying failure mechanism is revealed in our discordant pair analysis:
* On the 26 questions where Dense and BM25 produced divergent rankings, **Dense defeated BM25 on 20 questions (76.9%)**, while BM25 defeated Dense on only **6 questions (23.1%)**.
* Reciprocal Rank Fusion computes rank consensus across all queries without evaluating retriever reliability. Because `gemini-embedding-001` handles policy codes and numbers relatively well, BM25 adds marginal value while actively pulling down clean rankings across `multi_hop`, `aggregation`, and `paraphrase` queries.

---

## 5. Part C — Two-Stage Reranking

We evaluated two-stage retrieval: broad first-stage retrieval ($k=30$) followed by precision reranking down to the top 5 candidates.

### 5.1 C1: Cross-Encoder Evaluation (`ms-marco-MiniLM-L-6-v2`)

| Configuration | hit_rate@1 | hit_rate@5 | recall@5 | MRR | nDCG@5 | Latency p95 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| First-Stage Dense Top 5 | **0.7857** | 0.9762 | **0.8909** | **0.8770** | **0.8313** | **3.04 ms** |
| Dense Top 30 + Cross-Encoder | 0.7619 | **1.0000** | 0.8889 | 0.8619 | 0.8174 | 1,137.96 ms |
| **Delta ($\Delta$)** | **-0.0238** | **+0.0238** | **-0.0020** | **-0.0151** | **-0.0139** | **+1,134.9 ms** |

*Root Cause of Cross-Encoder Failure:* `ms-marco-MiniLM-L-6-v2` is pre-trained on open-domain web search queries (Bing queries paired with short web snippets). Insurance contracts consist of dense, conditional regulatory prose with severe legal distractor penalties. The cross-encoder suffered from **domain distribution shift**, frequently favoring conversational distractor passages over precise regulatory clauses.

### 5.2 C2: LLM Reranker Profiling (`gemini-3.5-flash-lite`)
* **Cost Accounting:** Evaluating 30 candidate passages per query consumes $\sim 7,500$ input tokens and $\sim 60$ output tokens per query. At `gemini-3.5-flash-lite` pricing ($0.075 / 1M input tokens), the marginal cost is **$0.56 per 1,000 queries**.
* **Latency Profile:** Executed sequentially, 30 LLM calls require $\sim 28.0$ seconds per query. When rate-limited by free-tier quotas, sequential execution throttles to **176.9 seconds/query**.
* **Quality Ceiling:** In batch benchmarks, the LLM reranker achieves **nDCG@5 = 0.8602** and **hit_rate@1 = 0.8095**, setting the absolute quality upper bound in the lab.

### 5.3 C3: The Reranking Decision Matrix & Dual Deployment Strategy

| Configuration | nDCG@5 | hit_rate@1 | Latency (p95) | Cost / 1k Queries |
|:---|:---:|:---:|:---:|:---:|
| **Dense (Un-reranked)** | 0.8313 | 0.7857 | **3.04 ms** | **$0.00** |
| **Dense + Cross-Encoder** | 0.8174 | 0.7619 | 1,137.96 ms | $0.00 |
| **Dense + LLM Reranker** | **0.8602** | **0.8095** | 28,000 ms (seq) | $0.56 |

#### The Two Deployment Decisions

1. **Deployment A: Interactive Agent-Facing Search Box**
   * **Recommendation:** **Deploy Dense (Un-reranked) on `markdown-400`.**
   * **Justification:** Support agents require sub-second UI responsiveness while interacting live with customers. Dense retrieval delivers a p95 latency of **3.04 ms** at zero marginal inference cost, while delivering 0.8313 nDCG@5 and 0.7857 hit_rate@1. Cross-Encoder reranking is eliminated because it lowers quality (-1.4 points) while introducing a 1.1-second lag. The LLM reranker's 28-second response time violates interactive UX latency budgets.
2. **Deployment B: Overnight Batch Auditing / Re-indexing Job**
   * **Recommendation:** **Deploy Dense + LLM Reranker (parallelized).**
   * **Justification:** In an asynchronous overnight audit pipeline processing historical claims, per-query latency is unconstrained. The system can parallelize candidate scoring across asynchronous worker pools (reducing wall-clock latency to $\sim 1.2$ seconds). At an operational cost of only **$0.56 per 1,000 queries**, capturing an additional +2.9 nDCG@5 points and +2.4% top-1 precision justifies the marginal expense.

### 5.4 C4: Failure Mode 5 Case Study (Reranker Inversion)
* **Question Q01:** *"How many days do I have to submit a reimbursement claim after discharge?"*
* **Target Docs:** `claims-timelines`, `claims-process`
* **First-Stage Dense Result:** Rank #1 = `claims-timelines` (**MRR = 1.0000**).
* **Post-Cross-Encoder Result:** Rank #1 = `claims-process`, Rank #2 = `claims-timelines` (**MRR = 0.5000**).
* **Diagnosis:** The first-stage retriever successfully surfaced the primary timeline schedule. The cross-encoder assigned a higher relevance score to the general procedural narrative in `claims-process` due to web-search keyword frequency, demoting the exact quantitative schedule to rank 2.

---

## 6. Part D — Index Scale & Metadata Engineering

### 6.1 D1: Exact NumPy vs. ChromaDB (HNSW)

| Index Implementation | Algorithm | hit_rate@1 | hit_rate@5 | recall@5 | MRR | nDCG@10 | Latency p95 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **NumPy (`DenseRetriever`)** | Exact Brute-Force Dot Product | **0.7857** | **0.9762** | **0.9028** | **0.8800** | **0.8527** | **2.09 ms** |
| **ChromaDB (`ChromaRetriever`)** | Approximate HNSW Graph | **0.7857** | **0.9762** | **0.9028** | **0.8800** | **0.8527** | 6.11 ms |

*Finding:* At corpus scale ($N=235$), the recall gap between exact search and HNSW is exactly **0.0000**. However, exact search is **nearly 3× faster** due to lower framework overhead.

### 6.2 D2: Search Latency Scaling & The Crossover Benchmark

We benchmarked exact matrix multiplication against Chroma HNSW across three orders of magnitude:

| Scale ($N$ Chunks) | Exact NumPy Latency (p50) | Chroma HNSW Latency (p50) | Speed Ratio (HNSW / NumPy) | Operational Winner |
|:---:|:---:|:---:|:---:|:---|
| **235** | **0.069 ms** | 1.299 ms | **18.9× slower** | Exact NumPy |
| **4,000** | **1.710 ms** | 4.116 ms | **2.4× slower** | Exact NumPy |
| **40,000** | 19.289 ms | **5.321 ms** | **3.6× faster** | **Chroma HNSW** |

```
Query Latency (ms)
 20.0 ┼                                         ● (NumPy: 19.29 ms)
 15.0 ┼
 10.0 ┼                      Crossover (~12k)
  5.0 ┼                                  ┌──────● (HNSW: 5.32 ms)
      ┼──● (NumPy: 0.07ms)───────────────┘
  0.0 ┴──────────────────────────────────────────────────────▶ Scale (Chunks)
      235                      4,000                    40,000
```

#### Hardware & Algorithmic Mechanism
* **Why HNSW loses at small scale:** Exact search utilizes BLAS Level-3 matrix routines executing SIMD vector instructions on contiguous L1/L2 CPU cache memory. HNSW requires traversing a multi-layered graph in memory, incurring pointer indirection, cache misses, and Python-to-C++ binding overhead ($\sim 1.0$ ms constant overhead).
* **The Crossover:** Exact search scales linearly with corpus size, $\mathcal{O}(N \cdot d)$. HNSW scales logarithmically, $\mathcal{O}(\log N)$. As corpus size crosses $\sim 12,000$ chunks, the compute cost of linear matrix multiplication surpasses graph traversal overhead, making HNSW asymptotically superior.

### 6.3 D3: The Archive Trap & Metadata Filtering (Q29–Q31)
Questions Q29, Q30, and Q31 test temporal policy conflicts where valid 2026 rules contradict obsolete rules in `claims-timelines-2024-ARCHIVED`.

* **Ingest-Time Tagging:** Each chunk is tagged with structured metadata: `status = "archived" if "ARCHIVED" in doc_id else "current"`.
* **Query-Time Enforcement:** Chroma search executed with `where={"status": "current"}`:

| Trap Question | Target Gold Documents | Unfiltered Top Retrieval | Filtered Top Retrieval | Unfiltered hit@1 | Filtered hit@1 |
|:---|:---|:---|:---|:---:|:---:|
| **Q29** | `claims-timelines` | `claims-timelines` | `claims-timelines` | 1.0 | 1.0 |
| **Q30** | `claims-process`, `claims-timelines` | `claims-timelines-2024-ARCHIVED` | `claims-timelines` | **0.0** | **1.0** |
| **Q31** | `claims-timelines`, `claims-process` | `claims-process` | `claims-process` | 1.0 | 1.0 |
| **Trap Group Mean** | — | — | — | **0.667** | **1.000** |

> ### The Central Architectural Takeaway
> **This critical performance improvement required zero changes to embedding vectors, similarity metrics, or retriever code.**
> In enterprise RAG, high error rates are frequently treated as embedding model failures. In reality, obsolete, conflicting, or out-of-scope documents in the index cannot be resolved by semantic similarity alone. **The most effective retrieval optimization is often metadata filtering at ingest, not retrieval engineering.**

---

## 7. Rubric Compliance & Target Scorecard

All performance targets specified in [`labs/lab3/README.md`](file:///c:/Users/Acer/Desktop/AI-in-Practice-Lab/aip-lab1/labs/lab3/README.md) were fully achieved or exceeded:

| Evaluation Metric | Target Threshold | Provided BM25 Baseline | Reference Solution | Our Measured Final Configuration | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **nDCG@10** | $\ge 0.800$ | 0.7010 | 0.8530 | **0.8527** | **Met** |
| **recall@5** | $\ge 0.850$ | 0.7900 | 0.9030 | **0.9028** | **Met** |
| **hit_rate@1** | $\ge 0.650$ | 0.5240 | 0.7860 | **0.7857** | **Met** |
| **MRR (`paraphrase` subset)** | $\ge 0.750$ | 0.5000 | 0.9000 | **0.8000** | **Met** |
| **Retrieval Latency (p95)** | $\le 400\text{ ms}$ | 0.30 ms | 1.00 ms | **2.78 ms** | **Met** |
| **Index Build Cost** | Reported | $0.00 | $0.008 | **$0.008** (cached) | **Met** |

---

## 8. Summary of Surprises & Lessons Learned

1. **Hybrid Retrieval is Not a Silver Bullet:** Despite being widely recommended in textbooks, RRF hybrid retrieval degraded overall performance on our corpus (nDCG@10 fell from 0.8527 to 0.7949). Published best practice is an empirical prior, not an operational mandate.
2. **Cross-Encoders Suffer From Severe Domain Shift:** The off-the-shelf cross-encoder lowered retrieval quality while adding over 1,100 ms of latency. Rerankers are neural models with specific training distributions that must be calibrated to domain prose.
3. **Approximations Must Earn Their Overhead:** Deploying vector databases (HNSW) prematurely on corpora smaller than $\sim 10,000$ chunks imposes an 18× latency penalty with zero quality benefit. Exact NumPy matrix dot products are strictly superior up to 10k–15k passages.
4. **Ingest Metadata Beats Algorithmic Tuning:** The 0.667 to 1.000 accuracy jump on temporal trap questions required zero algorithmic tuning—only metadata labeling and filtering at ingest.
