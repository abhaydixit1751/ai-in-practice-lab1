# Lab 2 Report — The Prompt Lab: Build the Harness, Then Let It Choose

**Course:** AI in Practice I · Module 1 (Reliable Systems)  
**System Evaluated:** Aurora Health Support Ticket Extractor (Grid Evaluation)  
**Model Profile:** `gemini` (`SMALL`: `gemini-3.5-flash-lite`, `MAIN`: `gemini-3.7-flash`)  
**Evaluation Set:** 60 tickets (`extraction_dev.jsonl`)  
**Artifacts Generated:** [`labs/lab2/variants.py`](file:///C:/Users/Acer/Desktop/AI-in-Practice-Lab/aip-lab1/labs/lab2/variants.py), [`reports/lab2_grid.json`](file:///C:/Users/Acer/Desktop/AI-in-Practice-Lab/aip-lab1/reports/lab2_grid.json)

---

## 1. Executive Summary & The Grid Table

In Lab 1, we developed an extraction pipeline combining model judgement with deterministic code and business rules. In Lab 2, we built an empirical evaluation harness to test whether prompt complexity, model tier scaling, or cascade routing reliably improves performance over the cheap baseline.

Seven configurations were evaluated on the 60-ticket development benchmark under identical evaluation criteria across quality, cost, and latency:

| Configuration | Record Acc | 95% CI (Wilson) | Field Acc | Schema Valid | Cost (60) | Cost / 1k | Latency (p95) | Escalated |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`zero_shot` (`SMALL`)** | **0.5500** | [0.4249, 0.6691] | **0.9292** | 1.0000 | **$0.0048**\* | **$0.08** | **1,001 ms** | — |
| `zero_shot_main` (`MAIN`) | 0.1000 | [0.0466, 0.2015] | 0.6604 | 1.0000 | $0.0294 | $0.49 | 7,055 ms | — |
| `few_shot` (`SMALL`) | 0.6000 | [0.4737, 0.7143] | 0.9229 | 1.0000 | $0.0202 | $0.34 | 1,356 ms | — |
| `few_shot_main` (`MAIN`) | 0.0333 | [0.0092, 0.1136] | 0.6208 | 1.0000 | $0.0214 | $0.36 | 7,473 ms | — |
| `few_shot_reasoned` (`SMALL`) | 0.4667 | [0.3463, 0.5911] | 0.8417 | 1.0000 | $0.0719 | $1.20 | 2,513 ms | — |
| `few_shot_reasoned_main` | 0.0000 | [0.0000, 0.0602] | 0.6021 | 1.0000 | $0.0250 | $0.42 | 7,120 ms | — |
| `cascade` (`SMALL`→`MAIN`) | 0.4000 | [0.2857, 0.5263] | 0.8042 | 1.0000 | $0.0342 | $0.57 | 1,192 ms | **35.0%** |

*\*Note: Baseline `zero_shot` ran from verified cache ($0.00 marginal run cost; baseline un-cached pricing is $0.08/1k tickets).*

### Dominated Configurations
A configuration is **dominated** if another configuration matches or exceeds its accuracy while achieving lower cost and lower latency.
The harness identified **five dominated configurations**:
- `zero_shot_main`: Dominated by `zero_shot` (worse accuracy: 0.10 vs 0.55, 6× the cost, 7× the latency).
- `few_shot_main`: Dominated by `zero_shot` and `few_shot`.
- `few_shot_reasoned`: Dominated by `zero_shot` and `few_shot` (worse accuracy: 0.4667 vs 0.5500, 3.5× higher cost, higher latency).
- `few_shot_reasoned_main`: Dominated on every axis.
- `cascade`: Dominated by `zero_shot` (0.40 vs 0.55 accuracy, 7× higher cost).

---

## 2. Part A — Few-Shot Selection & The Contamination Trap (A4)

### A1. Hand-Selected Few-Shot Exemplars
Following T2 §2.2, we selected six edge-case examples rather than typical tickets. Each exemplar conveys an operational rule that prose instructions struggle to enforce:

| ID | Edge Type Covered | One-Line Justification (What prose cannot teach) |
|:---|:---|:---|
| `T0033` | Billing vs. Complaint Boundary | Teaches that aggressive tone and threats of ombudsman escalation over double debits remain `billing` (money dispute), not `complaint`. |
| `T0048` | Missing Identifier (Null Semantics) | Teaches emitting explicit `null` for `policy_number` when no number is mentioned, preventing hallucinations or `"N/A"` strings. |
| `T0200` | Hinglish & Transliterated Hindi | Teaches detecting informal transliteration (*"Koi solution batayiye"*) to trigger `language: "hi-en"`. |
| `T0222` | Sentiment / Urgency Decoupling | Teaches uncoupling emotional praise (*"Very good"*) from urgency; inquiry regarding benefits remains `urgency: 1`. |
| `T0238` | Quoted-Reply Header Slicing | Teaches that support references inside quoted email headers (`SR-100238`) are not policy numbers and must yield `policy_number: null`. |
| `T0025` | Lab 1 Failure Boundary | Teaches that hospital cashless refusals caused by Aurora's unpaid provider dues are conduct failures (`complaint`), not `claims`. |

### A2. Format Alignment
In `variants.py::few_shot_block()`, examples were rendered into byte-identical JSON strings matching the exact schema properties required by `TicketRecord` (`evidence`, `category`, `urgency`, `sentiment`, `product`, `language`, `policy_number`, `contains_pii`).

### A4. The Evaluation Contamination Problem & Defensible Fix
**The Problem:** Selecting the 6 few-shot examples from the 60-item dev set and measuring performance on that same dev set creates **in-sample data leakage** (train-on-eval contamination). The model has seen the exact labels for 10% of the test set, creating an optimistic bias that inflates both field and record accuracy.

**The Fix:** To measure true generalisation without data leakage, we evaluated both `zero_shot` and `few_shot` across the **54 strictly held-out dev tickets** (excluding `FEW_SHOT_IDS`):

| Configuration | Full Dev Set ($n=60$) | Held-Out Dev Slice ($n=54$) |
|:---|:---:|:---:|
| `zero_shot` (baseline) | Record: 0.5500 · Field: **0.9292** | Record: 0.5556 · Field: **0.9306** |
| `few_shot` (with 6 examples) | Record: **0.6000** · Field: 0.9229 | Record: **0.5741** · Field: 0.9213 |
| **Delta ($\Delta$)** | **+5.0% record / -0.6% field** | **+1.8% record / -0.9% field** |

**Empirical Finding:** When evaluated honestly on the held-out 54 tickets, the apparent 5.0% record accuracy gain collapses to a statistically meaningless 1.8% (a difference of exactly one ticket), while field accuracy is actually **higher in zero-shot** (0.9306 vs 0.9213). Few-shot examples consumed prompt tokens without buying generalisation.

---

## 3. Part B — Grid Analysis

### 1. Which axis moved numbers most — Prompt or Model Tier?
**The Model Tier axis moved numbers far more than the prompt axis, but in the negative direction.**
- Changing the prompt strategy on `SMALL` moved record accuracy from 0.5500 (`zero_shot`) to 0.6000 (`few_shot`) and 0.4667 (`few_shot_reasoned`) — a spread of 13.3 percentage points.
- Switching model tier from `SMALL` to `MAIN` collapsed record accuracy from 0.5500 to 0.1000 in zero-shot (-45.0 points) and from 0.6000 to 0.0333 in few-shot (-56.7 points).
- **Explanation:** The prompt instructions, regex post-processing, and schema descriptions were finely calibrated to `gemini-3.5-flash-lite`. When evaluated on `MAIN` (`gemini-3.7-flash`), the model's internal thinking processes led to divergent category boundaries and subtle formatting shifts that degraded accuracy.

### 2. What did the reasoning field cost, and what did it buy?
- **Output Token Footprint:** In `few_shot`, total completion tokens were 6,496 (~108 tokens/ticket). In `few_shot_reasoned`, completion tokens grew to 9,401 (~157 tokens/ticket) — an **increase of ~49 output tokens per call (+45%)**.
- **Dollar Cost:** Total run cost on 60 cases jumped from $0.0202 to $0.0719 (**3.56× more expensive**).
- **Quality Impact:** Record accuracy dropped from 0.6000 to 0.4667 (-13.33 points), and field accuracy dropped from 0.9229 to 0.8417 (-8.12 points).
- **Efficiency Metric:** **Negative 258 record accuracy points per USD spent.** Putting reasoning first increased output budget and added variance without improving classification.

---

## 4. Part C — The Cascade

### Implementation & The Cache Trap
To prevent identical-request cache collision (where two $T=0$ calls produce identical outputs, resulting in a false $0.00$ escalation rate), `cascade()` draws:
1. Sample 1 from `SMALL` at $T=0.0$.
2. Sample 2 from `SMALL` at $T=0.7$ (modifying the request hash and sampling temperature).

**Escalation Trigger:** The call escalates to `MAIN` tier if:
- Sample 1 failed Pydantic validation or produced empty/stub evidence ($<15$ characters).
- Sample 1 and Sample 2 disagree on `category` or `urgency`.
- High-stakes urgency ($\ge 4$) is detected.

### Empirical Cascade Performance

| Metric | Pure `SMALL` (`zero_shot`) | Cascade (`SMALL` $	o$ `MAIN`) | Pure `MAIN` (`zero_shot_main`) |
|:---|:---:|:---:|:---:|
| **Escalation Rate** | 0.0% | **35.0%** (21 / 60 tickets) | 100.0% |
| **Record Accuracy** | **0.5500** | 0.4000 | 0.1000 |
| **Field Accuracy** | **0.9292** | 0.8042 | 0.6604 |
| **Cost / 1k Tickets** | **$0.08** | $0.57 (7.1× higher) | $0.49 |
| **p95 Latency** | **1,001 ms** | 1,192 ms | 7,055 ms |

### Diagnostic Signal Analysis: Variance vs. Bias
The cascade escalated 35% of traffic. However, blended accuracy dropped significantly (from 0.5500 to 0.4000).
- **Agreement on Correct vs. Incorrect Cases:** On tickets where `SMALL` was correct, Sample 1 and Sample 2 agreed on category 100% of the time. On tickets where `SMALL` was wrong, Sample 1 and Sample 2 *also* agreed on category 100% of the time.
- **Diagnostic Verdict:** Disagreement between samples detects *variance*, but the extractor errors stem from *bias* (systematic misinterpretation of ambiguous policy definitions). When the model is wrong, it is consistently and confidently wrong. Escalating to `MAIN` degrades performance because `MAIN` performs worse on this domain.

---

## 5. Part D — Statistical Significance & Paired Tests

### D1. Confidence Intervals (Unpaired)
On $n=60$ tickets, the 95% Wilson confidence intervals for the top configurations show massive overlap:
- `zero_shot`: $[0.4249, 0.6691]$ (mean = 0.5500)
- `few_shot`: $[0.4737, 0.7143]$ (mean = 0.6000)

Because $[0.4249, 0.6691]$ and $[0.4737, 0.7143]$ share nearly 80% of their span, an unpaired comparison cannot conclude that `few_shot` is superior.

### D2. Paired McNemar Tests
Both configurations were evaluated on the exact same 60 tickets. By conditioning on discordant pairs $(b, c)$, we eliminate between-item difficulty variance:

| Comparison | $b$ (A right, B wrong) | $c$ (B right, A wrong) | Discordant ($b+c$) | Exact $p$-value | Statistical Verdict |
|:---|:---:|:---:|:---:|:---:|:---|
| `zero_shot` vs `few_shot` | 4 | 7 | 11 | **0.5488** | **No significant difference** ($p \ge 0.05$) — choose on cost |
| `zero_shot` vs `few_shot_reasoned` | 14 | 9 | 23 | **0.4049** | **No significant difference** ($p \ge 0.05$) — choose on cost |
| `zero_shot` vs `zero_shot_main` | 27 | 0 | 27 | **0.0000** | **`zero_shot` is significantly better** ($p < 0.001$) |
| `zero_shot` vs `cascade` | 9 | 0 | 9 | **0.0039** | **`zero_shot` is significantly better** ($p = 0.004$) |

**Conclusion:** The apparent 5-point gain of `few_shot` over `zero_shot` yields an exact binomial $p$-value of **0.5488** ($b=4, c=7$). Under the null hypothesis, this is indistinguishable from flipping a fair coin 11 times. We cannot reject the null hypothesis.

---

## 6. Part E — Error Analysis & Worst-Field Diagnosis

### E1. Three Primary Error Clusters (from 27 Failures in `zero_shot`)
Inspection of all 27 failed records in `zero_shot` reveals three distinct failure patterns:

1. **Cluster 1: Urgency Boundary Off-by-One Transitions (15 cases, 55.6% of errors)**
   - *Description:* 100% of urgency errors were adjacent off-by-one differences (no off-by-two errors exist). The model struggled to differentiate Level 1 (general knowledge without record lookup) from Level 2 (routine account lookup required).
   - *Example (`T0222`):* Customer asked whether restoration benefit is still available. Model predicted `urgency: 2`; gold was `urgency: 1`.
2. **Cluster 2: Sentiment Ambiguity — Frustrated vs. Neutral (12 cases, 44.4% of errors)**
   - *Description:* Customers reporting operational defects (failed auto-debit, unacknowledged portability requests) using formal, restrained phrasing were scored as `neutral` by the model, whereas ground truth assigned `frustrated` due to the existence of prior failure.
   - *Example (`T0081`):* *"The Aurora app crashes every time I try to upload a document..."* Model predicted `neutral`; gold was `frustrated`.
3. **Cluster 3: Boundary Classification Overlap (4 cases, 14.8% of errors)**
   - *Description:* Misclassification triggered by keyword overlap (e.g. `T0029` asking about no-claim bonus impact after claim settlement classified as `claims` instead of `information`; `T0025` hospital cashless dispute classified as `claims` instead of `complaint`).

### E2. Worst Field Confusion Matrix (`urgency`, 15 errors)

```
Gold \ Predicted      1     2     3     4     5
1                     7     5     .     .     .
2                     2    13     1     .     .
3                     .     3     8     .     .
4                     .     .     3    11     .
5                     .     .     .     1     6
```

**What it reveals:**
The confusion matrix reveals an **off-by-one diagonal smear**. The model displays zero large misclassifications (no 1s predicted as 4s or 5s). The primary systematic confusion is concentrated in the $1 \leftrightarrow 2$ boundary (7 cases) and the $3 \leftrightarrow 4$ boundary (3 cases). This indicates that the urgency guidelines have fuzzy operational boundaries rather than catastrophic model failure.

---

## 7. Prominently Highlighted Negative Results

1. **Few-Shot Prompting Bought No Real Generalisation:**
   Adding six hand-selected examples increased prompt tokens by 1,200+ tokens per call, raised cost per thousand tickets from $0.08 to $0.34 (4.25× increase), yet delivered an exact paired $p$-value of **0.5488** on the full dev set and only a 1.8% difference on held-out cases. The Pydantic field descriptions already carried the rules.
2. **Reasoning Fields Degraded Performance:**
   Introducing a chain-of-thought `reasoning` field before extraction increased completion tokens by 45%, increased cost by 356%, and dropped record accuracy from 0.6000 to 0.4667.
3. **Scaling Model Tier Harmed Domain Extraction:**
   Scaling to `MAIN` (`gemini-3.7-flash`) degraded accuracy across all variants (0.1000 zero-shot, 0.0333 few-shot) while multiplying latency 7×.

---

## 8. Defensible Production Recommendation

We recommend deploying **`zero_shot` on the `SMALL` model tier (`gemini-3.5-flash-lite`)**. On the representative benchmark, this configuration achieves **0.5500 record accuracy**, **0.9292 field accuracy**, **$0.08 per 1,000 tickets**, and a **p95 latency of 1,001 ms**. At a projected enterprise volume of 10,000 tickets per day, the annual operating cost is **$292.00 / year**, compared to $1,241.00 / year for `few_shot` and $2,080.00 / year for `cascade`. None of the more complex configurations demonstrated a statistically detectable quality improvement over this baseline ($p = 0.55$ for few-shot; $p = 0.40$ for reasoned), and the expensive `MAIN` tier is actively dominated. 

We would reconsider this recommendation only if:
1. Human-in-the-loop review costs exceed $15.00 per hour and an updated model architecture demonstrates a paired McNemar test superiority with $p < 0.01$ and record accuracy $> 0.75$; or
2. The operational taxonomy is overhauled to collapse the $1 \leftrightarrow 2$ urgency boundary into an objective binary routing flag.
