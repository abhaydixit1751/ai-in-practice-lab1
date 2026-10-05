#!/usr/bin/env python3
"""Lab 4 — your RAG pipeline.

Write this yourself. `aip/rag.py` is the reference implementation; look at it
after Part A, not before. Labs 5-7 build on whichever of the two you prefer,
but you must be able to explain every line of the one you use.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.guards import UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted  # noqa: E402
from aip.llm import chat  # noqa: E402
from aip.retrieval import Hit, Retriever, format_context  # noqa: E402

# The exact string the system must emit when it cannot answer. Exact, because
# downstream code detects refusal by matching it -- a paraphrase is a bug.
REFUSAL = "I don't have enough information in the provided sources to answer that."

# Balanced (default) generation prompt meeting all 6 required elements + partial answers:
ANSWER_SYSTEM = f"""\
You answer questions using ONLY the numbered sources provided below.

Rules, in strict priority order:
1. Grounding: Answer strictly using facts from the numbered sources. Never rely on general knowledge or outside assumptions.
2. Refusal: If the provided sources do not contain enough information to answer the question, reply with the exact refusal string:
   "{REFUSAL}"
   Do not guess, assume, or invent details.
3. Partial Answers: If the sources only answer part of the question, state what is supported citing the sources, and explicitly state what cannot be answered from the sources using: "I don't have enough information in the provided sources to answer [the missing detail]."
4. Citations: Every factual sentence must end with a citation to the numbered source(s) supporting it, in the form [1] or [2][5].
5. Valid indices only: Never cite a number that was not provided in the context (cite only between 1 and the total number of sources).
6. Disagreements: If different sources disagree or contradict each other, explicitly state the contradiction and cite both conflicting sources.
7. Length discipline: Be concise. Keep your response to two or three sentences unless the question explicitly requires listing more.

{UNTRUSTED_SYSTEM_CLAUSE}
"""

# Stricter prompt for C4 refusal trade-off evaluation:
ANSWER_SYSTEM_STRICT = f"""\
You answer questions using ONLY the numbered sources provided below.

Rules, in strict priority order:
1. High-Precision Refusal: If the provided sources do not contain complete, definitive, and comprehensive information to answer EVERY aspect of the question, reply immediately with the exact refusal string:
   "{REFUSAL}"
   Do not provide partial answers or extrapolate.
2. Grounding: Answer strictly using facts from the numbered sources. Forbid general knowledge.
3. Citations: Every factual sentence must end with a citation in the form [1] or [2][5].
4. Valid indices only: Never cite a number not provided.
5. Disagreements: If sources disagree, surface the conflict and cite both.
6. Length discipline: Two or three sentences max.

{UNTRUSTED_SYSTEM_CLAUSE}
"""


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    citations_valid: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    n_citations: int = 0
    truncated: bool = False


def validate_answer(text: str, n_sources: int, finish_reason: str | None = None) -> dict:
    """B2. Validate model output against the RAG contract.

    Returns:
        {"valid": bool, "refused": bool, "invalid_citations": [ints],
         "n_citations": int, "truncated": bool, "reason": str}

    Checks:
      - every [n] is between 1 and n_sources
      - not truncated (finish_reason == "length")
      - a non-refusal answer contains at least one citation
    """
    truncated = (finish_reason == "length")
    clean = text.strip()

    cited = [int(m) for m in re.findall(r"\[(\d+)\]", clean)]
    invalid_citations = sorted({c for c in cited if c < 1 or c > n_sources})
    n_citations = len(cited)

    # Detect refusal: exact prefix or canonical refusal clause
    refused = (
        clean.startswith(REFUSAL[:40])
        or REFUSAL[:40].lower() in clean.lower()
    )

    if truncated:
        valid = False
        reason = "Answer was truncated by max_tokens limit (finish_reason == 'length')"
    elif not clean:
        valid = False
        reason = "Answer is empty"
    elif invalid_citations:
        valid = False
        reason = f"Citations {invalid_citations} are out of range (1..{n_sources})"
    elif not refused and n_citations == 0:
        valid = False
        reason = "Non-refusal answer contains no citations"
    else:
        valid = True
        reason = "ok"

    return {
        "valid": valid,
        "refused": refused,
        "invalid_citations": invalid_citations,
        "n_citations": n_citations,
        "truncated": truncated,
        "reason": reason,
    }


def _chat_with_retry(prompt: str, **kwargs):
    import time
    for attempt in range(6):
        try:
            return chat(prompt, **kwargs)
        except Exception as exc:
            if attempt == 5:
                raise
            err_str = str(exc).lower()
            if any(k in err_str for k in ("429", "quota", "rate", "resource_exhausted")):
                time.sleep(20.0)
            else:
                time.sleep(3.0)


def answer_question(question: str, retriever: Retriever, *, k: int = 12,
                    final_k: int = 5, reranker=None, tier: str = "SMALL",
                    system: str = ANSWER_SYSTEM) -> Answer:
    """Retrieve -> (rerank) -> generate -> validate -> maybe repair.

    B3 Policy:
    If validation fails (invalid citation, uncited claim, or truncation), we attempt
    a single corrective prompt explaining the exact validation failure. If the
    repaired answer still fails validation, we fall back to the safe REFUSAL string.
    This guarantees that the function NEVER returns an Answer with citations_valid=False
    and refused=False.
    """
    hits = retriever.search(question, k=k)
    if reranker is not None:
        hits = reranker.rerank(question, hits, k=final_k)
    else:
        hits = hits[:final_k]

    if not hits:
        return Answer(
            question=question,
            text=REFUSAL,
            hits=[],
            refused=True,
            citations_valid=True,
            invalid_citations=[],
            n_citations=0,
            truncated=False,
        )

    context = delimit_untrusted(format_context(hits, max_chars=8000))
    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer with citations:"

    res = _chat_with_retry(prompt, system=system, tier=tier, temperature=0.0, max_tokens=600, return_full=True)
    text = res["text"].strip()
    finish_reason = res.get("finish_reason")

    val = validate_answer(text, len(hits), finish_reason=finish_reason)

    # B3: Validation failure handling: retry once, then safe fallback
    if not val["valid"]:
        repair_prompt = (
            f"{prompt}\n\n"
            f"Your previous attempt: {text}\n\n"
            f"Validation error: {val['reason']}.\n"
            f"Please regenerate the answer adhering strictly to the contract:\n"
            f"- Cite only from sources [1] to [{len(hits)}].\n"
            f"- If the sources do not contain the answer, reply exactly: {REFUSAL}\n"
            f"- Every factual claim must carry a valid citation."
        )
        res_repair = _chat_with_retry(repair_prompt, system=system, tier=tier, temperature=0.0, max_tokens=600, return_full=True)
        text_repair = res_repair["text"].strip()
        val_repair = validate_answer(text_repair, len(hits), finish_reason=res_repair.get("finish_reason"))

        if val_repair["valid"]:
            text = text_repair
            val = val_repair
        else:
            # Fall back to refusal to ensure no ungrounded or bogus citation leaks to users
            text = REFUSAL
            val = validate_answer(text, len(hits), finish_reason=None)

    return Answer(
        question=question,
        text=text,
        hits=hits,
        refused=val["refused"],
        citations_valid=val["valid"],
        invalid_citations=val["invalid_citations"],
        n_citations=val["n_citations"],
        truncated=val["truncated"],
    )


def answer_with_gold_context(question: str, gold_docs: list[str], *,
                             tier: str = "SMALL", system: str = ANSWER_SYSTEM) -> Answer:
    """E2: Same generator, but the context is the gold documents.

    No retrieval at all -- format the gold documents, chunk them to match the
    generator's input structure, and generate. The difference between this and
    answer_question() is the damage retrieval is doing.
    """
    if not gold_docs:
        return Answer(
            question=question,
            text=REFUSAL,
            hits=[],
            refused=True,
            citations_valid=True,
            invalid_citations=[],
            n_citations=0,
            truncated=False,
        )

    chunks = []
    for i, doc_text in enumerate(gold_docs):
        chunks.extend(markdown_chunks(doc_text, f"gold_{i}", size=400))

    hits = [Hit(doc_id=c.doc_id, text=c.text, score=1.0) for c in chunks]

    context = delimit_untrusted(format_context(hits, max_chars=8000))
    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer with citations:"

    res = _chat_with_retry(prompt, system=system, tier=tier, temperature=0.0, max_tokens=600, return_full=True)
    text = res["text"].strip()
    finish_reason = res.get("finish_reason")

    val = validate_answer(text, len(hits), finish_reason=finish_reason)

    if not val["valid"]:
        repair_prompt = (
            f"{prompt}\n\n"
            f"Your previous attempt: {text}\n\n"
            f"Validation error: {val['reason']}.\n"
            f"Please regenerate the answer adhering strictly to the contract:\n"
            f"- Cite only from sources [1] to [{len(hits)}].\n"
            f"- If the sources do not contain the answer, reply exactly: {REFUSAL}\n"
            f"- Every factual claim must carry a valid citation."
        )
        res_repair = _chat_with_retry(repair_prompt, system=system, tier=tier, temperature=0.0, max_tokens=600, return_full=True)
        text_repair = res_repair["text"].strip()
        val_repair = validate_answer(text_repair, len(hits), finish_reason=res_repair.get("finish_reason"))

        if val_repair["valid"]:
            text = text_repair
            val = val_repair
        else:
            text = REFUSAL
            val = validate_answer(text, len(hits), finish_reason=None)

    return Answer(
        question=question,
        text=text,
        hits=hits,
        refused=val["refused"],
        citations_valid=val["valid"],
        invalid_citations=val["invalid_citations"],
        n_citations=val["n_citations"],
        truncated=val["truncated"],
    )

