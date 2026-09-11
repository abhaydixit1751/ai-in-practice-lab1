#!/usr/bin/env python3
"""Lab 2 — the configurations under test.

Each variant is a callable `str -> dict`. `grid.py` runs them all through the
same harness, so the only thing that differs between rows of your table is the
thing you intended to differ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from pydantic import Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.llm import structured  # noqa: E402
from labs.lab1.extract import (  # noqa: E402
    SYSTEM_PROMPT, TicketRecord, apply_business_rules, extract_deterministic,
)

# ---------------------------------------------------------------------------
# A1 — your six chosen examples.
# ---------------------------------------------------------------------------
# Six dev-set tickets covering key operational edge cases:
#   - T0033: billing/complaint boundary (hostile ombudsman escalation threat over double debit; still billing)
#   - T0048: ticket with no policy number (teaches emitting explicit null rather than hallucinating)
#   - T0200: Hinglish ticket ('Koi solution batayiye' -> language: hi-en)
#   - T0222: satisfied-but-urgent ticket (sentiment/urgency trap: praise with routine inquiry -> sentiment: satisfied, urgency: 1)
#   - T0238: quoted reply trap (SR-100238 in > quote header; live policy is null)
#   - T0025: Lab 1 failure case (hospital cashless refusal due to unpaid Aurora network dues is complaint, not claims)
FEW_SHOT_IDS: list[str] = [
    "T0033",  # teaches: billing/complaint boundary (double debit dispute with escalation threat is billing, not complaint)
    "T0048",  # teaches: absent policy number must be null (never guess or invent)
    "T0200",  # teaches: Hinglish code-mixing ('Koi solution batayiye') requires language='hi-en'
    "T0222",  # teaches: sentiment/urgency trap (satisfaction praise uncoupled from urgency 1 informational inquiry)
    "T0238",  # teaches: quoted reply history numbers (SR-100238) must be ignored, live policy is null
    "T0025",  # teaches: Lab 1 error; cashless denial caused by Aurora unpaid network hospital dues is complaint, not claims
]

FEW_SHOT_EXAMPLES: dict[str, dict] = {
    "T0033": {
        "evidence": "THIS IS THE THIRD TIME I am writing about the double debit on my policy.",
        "category": "billing",
        "urgency": 4,
        "sentiment": "angry",
        "product": "unknown",
        "language": "en",
        "policy_number": None,
        "contains_pii": True,
        "needs_human_review": False,
        "review_reason": "",
    },
    "T0048": {
        "evidence": "Please add my mother as a dependent on my policy.",
        "category": "policy_change",
        "urgency": 2,
        "sentiment": "neutral",
        "product": "unknown",
        "language": "en",
        "policy_number": None,
        "contains_pii": False,
        "needs_human_review": False,
        "review_reason": "",
    },
    "T0200": {
        "evidence": "My claim on AUR-9674338 was settled at Rs 41800 but the hospital bill was much higher.",
        "category": "claims",
        "urgency": 3,
        "sentiment": "frustrated",
        "product": "unknown",
        "language": "hi-en",
        "policy_number": "AUR-9674338",
        "contains_pii": True,
        "needs_human_review": False,
        "review_reason": "",
    },
    "T0222": {
        "evidence": "Can you confirm the restoration benefit is still available this year on AUR-4940770?",
        "category": "information",
        "urgency": 1,
        "sentiment": "satisfied",
        "product": "platinum",
        "language": "en",
        "policy_number": "AUR-4940770",
        "contains_pii": False,
        "needs_human_review": False,
        "review_reason": "",
    },
    "T0238": {
        "evidence": "I submitted a portability request 21 days ago and heard nothing.",
        "category": "policy_change",
        "urgency": 3,
        "sentiment": "frustrated",
        "product": "bronze",
        "language": "en",
        "policy_number": None,
        "contains_pii": True,
        "needs_human_review": False,
        "review_reason": "",
    },
    "T0025": {
        "evidence": "The network hospital refused cashless saying you have not settled their dues.",
        "category": "complaint",
        "urgency": 4,
        "sentiment": "frustrated",
        "product": "silver",
        "language": "en",
        "policy_number": "AUR-1589331",
        "contains_pii": True,
        "needs_human_review": False,
        "review_reason": "",
    },
}

FEW_SHOT_REASONING: dict[str, str] = {
    "T0033": "Customer demands refund for double debit with ombudsman escalation threat. The primary transaction is money taken twice, which is billing, not complaint. Urgency is 4 due to repeated failure ('THIRD TIME'). Aggressive escalation tone is angry. No policy number in live text. Phone number is PII.",
    "T0048": "Customer asks to add their 63-year-old mother as dependent on policy, which is altering contract (policy_change). Urgency is 2 for routine policy change. Sentiment is neutral matter-of-fact inquiry. No policy number is given, so policy_number is null. No PII.",
    "T0200": "Customer queries unexplained deduction on settled claim amount, which is claims. Urgency is 3 because customer is waiting for resolution on a stuck deduction. Sentiment is frustrated. AUR-9674338 is in live text. Mobile number is PII. 'Koi solution batayiye' is Hindi transliteration, so language is hi-en.",
    "T0222": "Customer praises quick approval ('Very good') and asks a general question about restoration benefit without a pending claim or dispute, which is information. Urgency is 1. Sentiment is satisfied. Aurora Platinum plan named. AUR-4940770 in live text. No PII.",
    "T0238": "Portability request submitted 21 days ago with no response, which is policy_change. Urgency is 3. Sentiment is frustrated referencing prior failure. Aurora Bronze plan named. The live message contains no AUR number; SR-100238 in quoted reply is a support ticket, not policy number, so policy_number is null. Mobile number is PII.",
    "T0025": "Hospital refused cashless admission because Aurora failed to settle network dues. Aurora's conduct and network failure is the primary issue, so category is complaint, not claims. Urgency is 4 (admission tomorrow). Sentiment is frustrated. AUR-1589331 in live text. Aurora Silver plan named. Phone and email are PII.",
}


def load_examples(ids: list[str]) -> list[dict]:
    rows = [json.loads(line) for line in
            (ROOT / "data/eval/extraction_dev.jsonl").open(encoding="utf-8") if line.strip()]
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown example ids: {missing}")
    return [by_id[i] for i in ids]


def few_shot_block(ids: list[str], include_reasoning: bool = False) -> str:
    """A2: render the examples into the prompt.

    The example output format is byte-identical to the JSON format requested
    from the model.
    """
    examples = load_examples(ids)
    blocks = []
    for ex in examples:
        eid = ex["id"]
        inp = ex["input"].strip()
        out_dict = dict(FEW_SHOT_EXAMPLES[eid])
        if include_reasoning:
            out_dict = {"reasoning": FEW_SHOT_REASONING[eid], **out_dict}
        json_str = json.dumps(out_dict, indent=2)
        blocks.append(f"Ticket:\n{inp}\n\nExtraction:\n{json_str}")
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# The variants
# ---------------------------------------------------------------------------
def zero_shot(ticket: str, tier: str = "SMALL") -> dict:
    """B: Lab 1 Part C baseline: model for judgement, code for deterministic."""
    try:
        model_rec = structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT, tier=tier)
        data = model_rec.model_dump()
    except Exception as exc:  # noqa: BLE001
        data = {
            "evidence": "",
            "category": "information",
            "urgency": 3,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "policy_number": None,
            "contains_pii": False,
            "needs_human_review": True,
            "review_reason": f"{type(exc).__name__}: {exc}",
        }
    data.update(extract_deterministic(ticket))
    data = apply_business_rules(data, ticket)
    return data


def few_shot(ticket: str, tier: str = "SMALL") -> dict:
    """B: zero_shot + the few-shot block."""
    prompt = (
        "Here are reference examples of correctly extracted tickets:\n\n"
        f"{few_shot_block(FEW_SHOT_IDS, include_reasoning=False)}\n\n---\n\n"
        f"Ticket to extract:\n{ticket}"
    )
    try:
        model_rec = structured(prompt, schema=TicketRecord, system=SYSTEM_PROMPT, tier=tier)
        data = model_rec.model_dump()
    except Exception as exc:  # noqa: BLE001
        data = {
            "evidence": "",
            "category": "information",
            "urgency": 3,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "policy_number": None,
            "contains_pii": False,
            "needs_human_review": True,
            "review_reason": f"{type(exc).__name__}: {exc}",
        }
    data.update(extract_deterministic(ticket))
    data = apply_business_rules(data, ticket)
    return data


class TicketRecordReasoned(TicketRecord):
    """B: add a `reasoning: str` field FIRST (T2 §3.3).

    Pydantic keeps declaration order, and field order in the JSON Schema
    influences generation order. Putting reasoning first makes it condition the
    answer; putting it last makes it a post-hoc rationalisation. You want the
    first. Measure the difference in output tokens.
    """
    reasoning: str = Field(
        description="Step-by-step reasoning analyzing ticket facts, categorization rules, urgency criteria, and sentiment before producing the extraction."
    )

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        props = schema.get("properties", {})
        if "reasoning" in props:
            ordered = {"reasoning": props["reasoning"]}
            for k, v in props.items():
                if k != "reasoning":
                    ordered[k] = v
            schema["properties"] = ordered
        return schema


def few_shot_reasoned(ticket: str, tier: str = "SMALL") -> dict:
    """B: few_shot with TicketRecordReasoned."""
    prompt = (
        "Here are reference examples of correctly extracted tickets with step-by-step reasoning:\n\n"
        f"{few_shot_block(FEW_SHOT_IDS, include_reasoning=True)}\n\n---\n\n"
        f"Ticket to extract:\n{ticket}"
    )
    try:
        model_rec = structured(prompt, schema=TicketRecordReasoned, system=SYSTEM_PROMPT, tier=tier)
        data = model_rec.model_dump()
    except Exception as exc:  # noqa: BLE001
        data = {
            "reasoning": "",
            "evidence": "",
            "category": "information",
            "urgency": 3,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "policy_number": None,
            "contains_pii": False,
            "needs_human_review": True,
            "review_reason": f"{type(exc).__name__}: {exc}",
        }
    data.update(extract_deterministic(ticket))
    data = apply_business_rules(data, ticket)
    return data


def cascade(ticket: str) -> dict:
    """C: SMALL first; escalate to MAIN on validation failure, short evidence,
    sample disagreement (T=0.7), or high urgency (>=4).

    Record which path each ticket took -- set rec['_path'] = 'small' | 'large'
    so grid.py can report the escalation rate.
    """
    path = "small"
    rec = None
    try:
        r1 = structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT, tier="SMALL")
        # Draw second sample at T=0.7 to change cache key and test self-consistency variance
        r2 = structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT, tier="SMALL", temperature=0.7)

        should_escalate = (
            len(r1.evidence.strip()) < 15
            or r1.urgency >= 4
            or r1.category != r2.category
            or r1.urgency != r2.urgency
        )
        if should_escalate:
            rm = structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT, tier="MAIN")
            rec = rm.model_dump()
            path = "large"
        else:
            rec = r1.model_dump()
    except Exception:  # noqa: BLE001
        try:
            rm = structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT, tier="MAIN")
            rec = rm.model_dump()
            path = "large"
        except Exception as exc:  # noqa: BLE001
            rec = {
                "evidence": "",
                "category": "information",
                "urgency": 3,
                "sentiment": "neutral",
                "product": "unknown",
                "language": "en",
                "policy_number": None,
                "contains_pii": False,
                "needs_human_review": True,
                "review_reason": f"{type(exc).__name__}: {exc}",
            }
            path = "large"

    rec["_path"] = path
    rec.update(extract_deterministic(ticket))
    rec = apply_business_rules(rec, ticket)
    return rec


VARIANTS = {
    "zero_shot": lambda t: zero_shot(t, "SMALL"),
    "zero_shot_main": lambda t: zero_shot(t, "MAIN"),
    "few_shot": lambda t: few_shot(t, "SMALL"),
    "few_shot_main": lambda t: few_shot(t, "MAIN"),
    "few_shot_reasoned": lambda t: few_shot_reasoned(t, "SMALL"),
    "few_shot_reasoned_main": lambda t: few_shot_reasoned(t, "MAIN"),
    "cascade": cascade,
}
