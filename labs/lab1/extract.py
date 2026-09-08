#!/usr/bin/env python3
"""Lab 1, Parts B and C — the extractor you actually ship.

Complete the TODOs. `run_eval.py` imports `extract_b` and `extract_c` from
here, so keep those two function names.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip.guards import _PII_PATTERNS
from aip.llm import StructuredOutputError, structured

CATEGORIES = Literal["billing", "claims", "policy_change",
                     "technical", "complaint", "information"]


# ===========================================================================
# PART B — the schema
# ===========================================================================
class TicketRecord(BaseModel):
    """The contract. Everything the model is allowed to say, and nothing else.

    Remember from T2 §3.2: field `description`s are shipped to the model as
    part of the JSON Schema. They are the highest-leverage place to put an
    instruction, because they sit next to the thing they govern. Write them as
    instructions to the model, not as documentation for a human.
    """

    # TODO B1a: `evidence` is declared BEFORE `category` so that generating
    # the verbatim citation acts as an autoregressive reasoning scratchpad (CoT in
    # schema, T2 §3.3) that conditions and improves category selection.
    evidence: str = Field(
        max_length=200,
        description="The span of the ticket that determined the category, quoted verbatim. One sentence at most."
    )

    category: CATEGORIES = Field(
        description="billing = money in (premiums, debits, double debits, refunds, invoices, 80D tax certs, instalment options). "
                    "claims = an actual or intended claim (cashless request, reimbursement docs/queries, settlement amount, deduction, rejection). "
                    "If customer asks what documents to submit or about claim status, it is claims, NOT information. "
                    "policy_change = altering contract (add/remove dependent or newborn, upgrade, port, change contact details). "
                    "If customer asks about adding a dependent or portability, it is policy_change, NOT information. "
                    "technical = app/portal/login/OTP/document upload is broken. "
                    "complaint = Aurora's conduct itself is the subject (mis-selling, hold times, staff rudeness, unaddressed grievances). "
                    "Note: An angry customer seeking resolution of their claim or billing dispute is claims or billing, NOT complaint. "
                    "information = general questions about policy terms or waiting periods with NO pending transaction, claim, policy change, or customer record lookup."
    )

    urgency: int = Field(
        ge=1, le=5,
        description="Urgency scale 1-5: "
                    "1 = general question answerable without opening any customer record (e.g. general cataract waiting period, how to download e-card); "
                    "2 = requires looking up this customer's account/policy (e.g. wellness points on policy, required docs for claim on policy), routine policy changes (adding dependent, portability), or app defect; "
                    "3 = something already went wrong or is stuck, customer waiting (e.g. debited twice, portability submitted weeks ago with no response, claim deduction query); "
                    "4 = repeated failure ('third time asking'), money/access at risk now, standing at hospital desk, or explicit escalation threat ('refund or I will go to ombudsman'); "
                    "5 = active emergency (e.g. ICU), critical denial demanding immediate reversal, or stating customer IS filing with ombudsman ('I am filing a complaint with the ombudsman'). "
                    "Add +1 (capped at 5) if message states same-day or next-morning deadline."
    )

    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description="Customer emotional tone: "
                    "angry = hostile, shouting, aggressive, or threatening; "
                    "frustrated = unhappy and tired of delays, referencing a prior failure or unanswered request, but civil; "
                    "neutral = matter-of-fact first-time request or query without reference to a prior failure; "
                    "satisfied = expresses gratitude, thanks, or praise."
    )

    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = Field(
        description="The Aurora health insurance plan explicitly named in the message: 'bronze', 'silver', 'gold', or 'platinum'. "
                    "Must be 'unknown' if no product name is explicitly stated in the text. Never infer from sum insured or context."
    )

    language: Literal["en", "hi-en"] = Field(
        description="'hi-en' if Hindi words, Hinglish phrases, or transliterated Hindi (e.g. kripya, jaldi, bahut, batayiye, karo) "
                    "appear anywhere in the text; 'en' if English only."
    )

    # Part B only: the model decides these. In Part C you will delete them
    # from this schema and compute them in code instead.
    policy_number: str | None = Field(
        default=None,
        pattern=r"^AUR-\d{7}$",
        description="Format AUR- followed by exactly 7 digits (e.g. AUR-1234567), copied verbatim from the live message body. "
                    "Must be null if no policy number appears in the live text. Never invent, guess, or reformat."
    )
    contains_pii: bool = Field(
        default=False,
        description="True if ticket contains a phone number or personal email address (excluding Aurora corporate emails "
                    "support@aurorahealth.example and grievance@aurorahealth.example). Personal names alone do not count."
    )

    # Set by our code, never by the model.
    needs_human_review: bool = Field(
        default=False,
        description="Always false in the output JSON. Reserved for internal error handling."
    )
    review_reason: str = Field(
        default="",
        description="Always empty string in the output JSON. Reserved for internal error handling."
    )

    @field_validator("policy_number", mode="before")
    @classmethod
    def _policy_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = str(v).strip()
        return None if v.lower() in {"", "null", "none", "n/a"} else v


SYSTEM_PROMPT = """\
You are an expert customer support extraction system for Aurora Health Insurance.
Your task is to accurately extract structured operational metadata from incoming customer tickets.

Follow these instructions strictly:
1. Examine the ticket carefully. Quote the decisive sentence into 'evidence' verbatim before choosing 'category'.
2. Classify 'category' using the exact rules in the schema. In particular, a ticket regarding a claim or billing dispute is 'claims' or 'billing' if the customer seeks resolution of that transaction; classify as 'complaint' only when Aurora's conduct or service quality itself is the primary issue.
3. Assign 'urgency' according to the anchored 1-5 scale. Judge the factual situation, not customer emotional volume.
4. Assess 'sentiment' independently of urgency. Reserve 'frustrated' for messages that cite a prior delay or failed attempt.
5. Identify 'product' only if an Aurora tier (bronze, silver, gold, platinum) is explicitly named. Otherwise use 'unknown'.
6. Detect 'language' as 'hi-en' if any Hindi vocabulary or Hinglish transliteration appears, otherwise 'en'.
7. For 'policy_number', extract only valid AUR-XXXXXXX from the unquoted message, or null if absent. Never hallucinate.
8. Set 'contains_pii' to true if phone numbers or customer personal emails are present.

Return ONLY a single valid JSON object adhering strictly to the provided JSON Schema.
"""


def extract_b(ticket: str) -> TicketRecord:
    """Part B: the model decides everything."""
    try:
        return structured(ticket, schema=TicketRecord, system=SYSTEM_PROMPT, tier="SMALL")
    except (StructuredOutputError, Exception) as exc:  # noqa: BLE001
        return TicketRecord(
            evidence="",
            category="information",
            urgency=3,
            sentiment="neutral",
            product="unknown",
            language="en",
            policy_number=None,
            contains_pii=False,
            needs_human_review=True,
            review_reason=f"{type(exc).__name__}: {exc}",
        )


# ===========================================================================
# PART C — move the deterministic work out of the model
# ===========================================================================
POLICY_RE = re.compile(r"\bAUR-\d{7}\b")

# The quoted-reply marker. Everything after this is history, not the current
# message. Part C3 asks you to decide what that means for policy extraction.
QUOTE_MARKER = re.compile(r"^\s*>", re.MULTILINE)


def extract_deterministic(ticket: str) -> dict:
    """Extract policy_number and contains_pii deterministically with zero model calls.

    policy_number:
        C3 trap rule: Strip quoted reply history (lines starting with '>') before
        searching for policy numbers. Lines beginning with '>' represent forwarded or
        historical email threads, which often carry stale or template numbers.
        Extract the first matching AUR-<7 digits> from the unquoted live message body.
        If no match exists in the live body, return None.

        Generalization note: This rule generalizes well to structured email/ticketing
        systems with standard blockquote indicators (RFC 3676 / markdown '>'), but
        in arbitrary unstructured email clients without angle brackets, quoted text
        parsing requires threading header heuristics (e.g. 'On ... wrote:').

    contains_pii:
        True if the ticket contains an Indian phone number or an email address that is
        not one of Aurora's published service addresses (support@aurorahealth.example,
        grievance@aurorahealth.example). Customer names alone do not count.
    """
    # Exclude quoted history for policy number: everything from the first '>' onward is history
    m = QUOTE_MARKER.search(ticket)
    live_body = ticket[:m.start()] if m else ticket
    pn_match = POLICY_RE.search(live_body)
    policy_number = pn_match.group(0) if pn_match else None

    # Check PII across the ticket
    has_phone = bool(_PII_PATTERNS["PHONE_IN"].search(ticket))
    emails = _PII_PATTERNS["EMAIL"].findall(ticket)
    aurora_emails = {
        "support@aurorahealth.example",
        "grievance@aurorahealth.example",
    }
    has_external_email = any(e.lower() not in aurora_emails for e in emails)
    contains_pii = has_phone or has_external_email

    return {
        "policy_number": policy_number,
        "contains_pii": contains_pii,
    }


def apply_business_rules(rec_fields: dict, ticket: str) -> dict:
    """Compute business rules in code.

    Rule:
        escalate = urgency >= 4 or 'ombudsman' appears in the ticket
    """
    out = dict(rec_fields)
    urgency = int(out.get("urgency", 1))
    out["escalate"] = urgency >= 4 or ("ombudsman" in ticket.lower())
    return out


class TicketRecordC(BaseModel):
    """The reduced schema the model sees in Part C.

    policy_number and contains_pii are removed because they are computed
    deterministically in code.
    """

    # evidence declared BEFORE category (scratchpad effect per T2 §3.3)
    evidence: str = Field(
        max_length=200,
        description="The span of the ticket that determined the category, quoted verbatim. One sentence at most."
    )

    category: CATEGORIES = Field(
        description="billing = money in (premiums, debits, double debits, refunds, invoices, 80D tax certs, instalment options). "
                    "claims = an actual or intended claim (cashless request, reimbursement docs/queries, settlement amount, deduction, rejection). "
                    "If customer asks what documents to submit or about claim status, it is claims, NOT information. "
                    "policy_change = altering contract (add/remove dependent or newborn, upgrade, port, change contact details). "
                    "If customer asks about adding a dependent or portability, it is policy_change, NOT information. "
                    "technical = app/portal/login/OTP/document upload is broken. "
                    "complaint = Aurora's conduct itself is the subject (mis-selling, hold times, staff rudeness, unaddressed grievances). "
                    "Note: An angry customer seeking resolution of their claim or billing dispute is claims or billing, NOT complaint. "
                    "information = general questions about policy terms or waiting periods with NO pending transaction, claim, policy change, or customer record lookup."
    )

    urgency: int = Field(
        ge=1, le=5,
        description="Urgency scale 1-5: "
                    "1 = general question answerable without opening any customer record (e.g. general cataract waiting period, how to download e-card); "
                    "2 = requires looking up this customer's account/policy (e.g. wellness points on policy, required docs for claim on policy), routine policy changes (adding dependent, portability), or app defect; "
                    "3 = something already went wrong or is stuck, customer waiting (e.g. debited twice, portability submitted weeks ago with no response, claim deduction query); "
                    "4 = repeated failure ('third time asking'), money/access at risk now, standing at hospital desk, or explicit escalation threat ('refund or I will go to ombudsman'); "
                    "5 = active emergency (e.g. ICU), critical denial demanding immediate reversal, or stating customer IS filing with ombudsman ('I am filing a complaint with the ombudsman'). "
                    "Add +1 (capped at 5) if message states same-day or next-morning deadline."
    )

    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description="Customer emotional tone: "
                    "angry = hostile, shouting, aggressive, or threatening; "
                    "frustrated = unhappy and tired of delays, referencing a prior failure or unanswered request, but civil; "
                    "neutral = matter-of-fact first-time request or query without reference to a prior failure; "
                    "satisfied = expresses gratitude, thanks, or praise."
    )

    product: Literal["bronze", "silver", "gold", "platinum", "unknown"] = Field(
        description="The Aurora health insurance plan explicitly named in the message: 'bronze', 'silver', 'gold', or 'platinum'. "
                    "Must be 'unknown' if no product name is explicitly stated in the text. Never infer from sum insured or context."
    )

    language: Literal["en", "hi-en"] = Field(
        description="'hi-en' if Hindi words, Hinglish phrases, or transliterated Hindi (e.g. kripya, jaldi, bahut, batayiye, karo) "
                    "appear anywhere in the text; 'en' if English only."
    )

    # Set by our code, never by the model.
    needs_human_review: bool = Field(
        default=False,
        description="Always false in the output JSON. Reserved for internal error handling."
    )
    review_reason: str = Field(
        default="",
        description="Always empty string in the output JSON. Reserved for internal error handling."
    )


SYSTEM_PROMPT_C = """\
You are an expert customer support extraction system for Aurora Health Insurance.
Your task is to accurately extract structured operational metadata from incoming customer tickets.

Follow these instructions strictly:
1. Examine the ticket carefully. Quote the decisive sentence into 'evidence' verbatim before choosing 'category'.
2. Classify 'category' using the exact rules in the schema. In particular, a ticket regarding a claim or billing dispute is 'claims' or 'billing' if the customer seeks resolution of that transaction; classify as 'complaint' only when Aurora's conduct or service quality itself is the primary issue.
3. Assign 'urgency' according to the anchored 1-5 scale. Judge the factual situation, not customer emotional volume.
4. Assess 'sentiment' independently of urgency. Reserve 'frustrated' for messages that cite a prior delay or failed attempt.
5. Identify 'product' only if an Aurora tier (bronze, silver, gold, platinum) is explicitly named. Otherwise use 'unknown'.
6. Detect 'language' as 'hi-en' if any Hindi vocabulary or Hinglish transliteration appears, otherwise 'en'.

Return ONLY a single valid JSON object adhering strictly to the provided JSON Schema.
"""


def extract_c(ticket: str) -> dict:
    """Part C: model for judgement, code for everything else.

    Returns a plain dict (model fields + deterministic fields + business rules)
    so that run_eval.py can score it against the gold labels directly.
    """
    try:
        model_rec = structured(ticket, schema=TicketRecordC, system=SYSTEM_PROMPT_C, tier="SMALL")
        data = model_rec.model_dump()
    except (StructuredOutputError, Exception) as exc:  # noqa: BLE001
        data = {
            "evidence": "",
            "category": "information",
            "urgency": 3,
            "sentiment": "neutral",
            "product": "unknown",
            "language": "en",
            "needs_human_review": True,
            "review_reason": f"{type(exc).__name__}: {exc}",
        }

    # Deterministic fields
    det = extract_deterministic(ticket)
    data.update(det)

    # Business rules
    data = apply_business_rules(data, ticket)
    return data


if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[2]
    sample = json.loads(
        (root / "data/eval/extraction_dev.jsonl").open(encoding="utf-8").readline()
    )
    print("--- ticket ---")
    print(sample["input"][:600])
    print("\n--- gold ---")
    print(sample["expected"])
    print("\n--- yours ---")
    print(extract_c(sample["input"]))
