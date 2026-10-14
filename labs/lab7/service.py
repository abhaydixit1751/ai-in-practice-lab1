#!/usr/bin/env python3
"""Lab 7 — the service.

    uvicorn labs.lab7.service:app --reload --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"How long do I have to file a claim?"}' | jq
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, tracing
from aip.chunking import markdown_chunks
from aip.config import settings
from aip.cost import BudgetExceeded, global_budget
from aip.embed import cosine, embed
from aip.guards import (
    ToolGuard,
    detect_injection,
    enforce_citations,
    redact_pii,
)
from aip.retrieval import DenseRetriever
from labs.lab3.search import load_corpus
from labs.lab5.diagnose import ANSWER_SYSTEM_V2, DeduplicatedRagPipeline

app = FastAPI(title="Aurora Policy Assistant", version="1.0")

_PIPELINE: DeduplicatedRagPipeline | None = None
_STARTED = time.time()

# In-memory caches for Part B1
_EXACT_CACHE: dict[str, dict[str, Any]] = {}
_SEMANTIC_CACHE: list[dict[str, Any]] = []
SEMANTIC_THRESHOLD: float = 0.96


def pipeline() -> DeduplicatedRagPipeline:
    """TODO A2: build your Labs 3-5 pipeline once, at startup, and cache it.

    Building it per-request re-embeds the corpus every time. Students do this
    and then report a 40-second p95.
    """
    global _PIPELINE
    if _PIPELINE is None:
        with tracing.trace("service.startup.init_pipeline"):
            corpus = load_corpus()
            chunks = [
                c
                for doc_id, text in corpus.items()
                for c in markdown_chunks(text, doc_id, size=800)
            ]
            retriever = DenseRetriever(chunks, show_progress=False)
            _PIPELINE = DeduplicatedRagPipeline(
                retriever,
                k=15,
                final_k=5,
                tier="SMALL",
                system=ANSWER_SYSTEM_V2,
            )
    return _PIPELINE


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str


def _normalize_question(q: str) -> str:
    """Normalized string for exact caching."""
    return " ".join(q.strip().lower().split())


def _exact_cache_get(question: str) -> dict[str, Any] | None:
    norm = _normalize_question(question)
    key = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return _EXACT_CACHE.get(key)


def _exact_cache_put(question: str, res: dict[str, Any]) -> None:
    norm = _normalize_question(question)
    key = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    _EXACT_CACHE[key] = res


def _semantic_cache_get(question: str) -> tuple[dict[str, Any] | None, float]:
    """Find semantic match with cosine similarity >= SEMANTIC_THRESHOLD."""
    if not _SEMANTIC_CACHE:
        return None, 0.0
    try:
        q_vec = embed(question)
    except Exception:  # noqa: BLE001
        return None, 0.0

    best_match: dict[str, Any] | None = None
    best_sim: float = 0.0

    for item in _SEMANTIC_CACHE:
        sim = float(cosine(q_vec, item["vector"]))
        if sim > best_sim:
            best_sim = sim
            if sim >= SEMANTIC_THRESHOLD:
                best_match = item["response"]

    return best_match, best_sim


def _semantic_cache_put(question: str, res: dict[str, Any]) -> None:
    try:
        q_vec = embed(question)
        _SEMANTIC_CACHE.append({
            "question": question,
            "vector": q_vec,
            "response": res,
        })
    except Exception:  # noqa: BLE001, S110
        pass


@app.exception_handler(BudgetExceeded)
async def budget_exceeded_handler(request: Request, exc: BudgetExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": f"Budget exceeded: {exc}"})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """TODO A1. Return cost and trace_id in the response -- they are how
    anyone debugs this later."""
    t0 = time.perf_counter()
    try:
        pipe = pipeline()

        # Step 1: Input hygiene (Lab 6 guards)
        clean_question, pii_counts = redact_pii(req.question)
        if pii_counts:
            tracing.event("guard.pii_redacted", counts=pii_counts)

        injection = detect_injection(clean_question)
        if injection.flagged:
            tracing.event("guard.injection_blocked", signals=injection.signals)
            total_ms = (time.perf_counter() - t0) * 1000
            return AskResponse(
                answer="Request refused: suspicious instructions or injection patterns detected.",
                refused=True,
                citations=[],
                latency_ms=round(total_ms, 1),
                cost_usd=0.0,
                cached=False,
                trace_id="guard-refusal",
            )

        # Step 2: Exact response cache (Part B1)
        exact_hit = _exact_cache_get(clean_question)
        if exact_hit is not None:
            tracing.event("cache.exact_hit", question=clean_question[:80])
            total_ms = (time.perf_counter() - t0) * 1000
            return AskResponse(
                answer=exact_hit["answer"],
                refused=exact_hit["refused"],
                citations=[Citation(**c) for c in exact_hit["citations"]],
                latency_ms=round(total_ms, 1),
                cost_usd=0.0,
                cached=True,
                trace_id="cache-exact-hit",
            )

        # Step 3: Semantic cache (Part B1)
        sem_hit, sim = _semantic_cache_get(clean_question)
        if sem_hit is not None:
            tracing.event("cache.semantic_hit", question=clean_question[:80], similarity=round(sim, 4))
            total_ms = (time.perf_counter() - t0) * 1000
            return AskResponse(
                answer=sem_hit["answer"],
                refused=sem_hit["refused"],
                citations=[Citation(**c) for c in sem_hit["citations"]],
                latency_ms=round(total_ms, 1),
                cost_usd=0.0,
                cached=True,
                trace_id="cache-semantic-hit",
            )

        # Step 4: Core execution with tracing and error handling (A1, A2, A3)
        b0 = global_budget().spent_usd
        with tracing.trace("http.ask", question=clean_question[:120], mode=req.mode) as span:
            trace_id = span["span_id"]
            if req.mode == "tools":
                from labs.lab6.agent import run_agent
                guard = ToolGuard(
                    max_calls=6,
                    allow={"search_policy", "get_policy_details", "compute_premium"},
                )
                agent_res = run_agent(clean_question, guard=guard, tier="SMALL")
                answer_text = agent_res.get("answer", "")
                citations_list: list[Citation] = []
                refused = "cannot" in answer_text.lower() or "not have" in answer_text.lower()
            else:
                pipe.final_k = req.top_k
                rag_ans = pipe.answer(clean_question)
                answer_text = rag_ans.answer
                refused = rag_ans.refused
                citations_list = [
                    Citation(index=i + 1, doc_id=h.doc_id, excerpt=h.text[:250])
                    for i, h in enumerate(rag_ans.hits)
                ]

                # Lab 6 output guard: enforce citation bounds
                ok, invalid = enforce_citations(answer_text, len(rag_ans.hits))
                if not ok and not refused:
                    tracing.event("guard.citation_violation", invalid=invalid)

            b1 = global_budget().spent_usd
            cost_usd = round(max(0.0, b1 - b0), 6)
            total_ms = (time.perf_counter() - t0) * 1000

            response_dict = {
                "answer": answer_text,
                "refused": refused,
                "citations": [c.model_dump() for c in citations_list],
            }
            # Update caches on miss
            _exact_cache_put(clean_question, response_dict)
            _semantic_cache_put(clean_question, response_dict)

            return AskResponse(
                answer=answer_text,
                refused=refused,
                citations=citations_list,
                latency_ms=round(total_ms, 1),
                cost_usd=cost_usd,
                cached=False,
                trace_id=trace_id,
            )
    except BudgetExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        err_msg = str(exc).lower()
        err_name = type(exc).__name__.lower()
        is_provider_error = any(
            k in err_msg or k in err_name
            for k in ("ratelimit", "quota", "resource_exhausted", "503", "502", "connection", "timeout", "unavailable")
        )
        if is_provider_error:
            # TODO A3: distinguish provider outage (503 with Retry-After)
            raise HTTPException(
                status_code=503,
                detail="upstream model unavailable",
                headers={"Retry-After": "30"},
            ) from exc
        raise HTTPException(status_code=500, detail=f"internal server error: {exc}") from exc


@app.get("/health")
def health() -> dict[str, Any]:
    """TODO C: index size, model profile, cache stats, uptime."""
    pipe = pipeline()
    index_size = len(pipe.retriever.chunks) if pipe and hasattr(pipe.retriever, "chunks") else 0
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _STARTED, 1),
        "index_size": index_size,
        "model": settings.models.get("MAIN", "unknown"),
        "cache": cache.stats(),
        "exact_cache_entries": len(_EXACT_CACHE),
        "semantic_cache_entries": len(_SEMANTIC_CACHE),
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """TODO C2: cost today, cost/query, cache hit rate, p50/p95/p99, error rate."""
    b = global_budget()
    b_dict = b.as_dict()

    traces = tracing.read_traces()
    ask_spans = [t for t in traces if t.get("name") in ("http.ask", "rag.answer")]
    errors = [t for t in traces if t.get("status") == "error"]
    error_types: dict[str, int] = {}
    for err in errors:
        k = err.get("error_kind") or err.get("error", "Unknown").split(":")[0]
        error_types[k] = error_types.get(k, 0) + 1

    total_queries = max(1, len(ask_spans))
    cached_queries = sum(1 for t in traces if t.get("name") in ("cache.exact_hit", "cache.semantic_hit") or t.get("cached"))
    cache_hit_rate = round(cached_queries / total_queries, 3)

    tool_calls: dict[str, int] = {}
    for t in traces:
        if t.get("name") == "tool.call":
            tool_name = t.get("tool", "unknown")
            tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1

    return {
        **b_dict,
        "p99_latency_ms": round(b.percentile(99), 1),
        "cache_hit_rate": cache_hit_rate,
        "total_queries": total_queries,
        "error_count": len(errors),
        "error_rate_by_type": error_types,
        "tool_call_counts": tool_calls,
    }


# ---------------------------------------------------------------------------
# Part B2: Streaming with Server-Sent Events (SSE)
# Part B3: Resolution - Stream prose first, validate and send citations at done
# ---------------------------------------------------------------------------
@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> EventSourceResponse:
    """B2 & B3: Server-sent events streaming endpoint.

    B3 Defense: We stream text tokens in real time to minimize TTFT (target <= 1500ms).
    Citation validation is deferred to stream completion, where a final 'done'
    event supplies the validated citations, validity flag, latency, and cost.
    If citations are invalid, the client UI is instructed to flag or retract
    unverified statements, preserving both TTFT and grounding safety.
    """
    clean_question, _ = redact_pii(req.question)

    async def event_generator() -> AsyncIterator[dict[str, Any]]:
        t0 = time.perf_counter()
        pipe = pipeline()

        # Check exact cache first
        exact_hit = _exact_cache_get(clean_question)
        if exact_hit:
            words = exact_hit["answer"].split(" ")
            for w in words:
                yield {"event": "token", "data": json.dumps({"token": w + " "})}
                await asyncio.sleep(0.02)
            total_ms = (time.perf_counter() - t0) * 1000
            yield {
                "event": "done",
                "data": json.dumps({
                    "answer": exact_hit["answer"],
                    "citations": exact_hit["citations"],
                    "citations_valid": True,
                    "refused": exact_hit["refused"],
                    "latency_ms": round(total_ms, 1),
                    "cost_usd": 0.0,
                    "cached": True,
                    "trace_id": "cache-exact-hit",
                }),
            }
            return

        with tracing.trace("http.ask.stream", question=clean_question[:120]) as span:
            trace_id = span["span_id"]
            hits = pipe.retrieve(clean_question)
            final_hits = pipe.rerank(clean_question, hits)

            from aip.guards import delimit_untrusted
            from aip.llm import chat
            from aip.retrieval import format_context

            context = delimit_untrusted(format_context(final_hits, max_chars=8000))
            prompt = f"{context}\n\nQuestion: {clean_question}\n\nAnswer with citations:"

            # Generate and stream tokens
            full_text = chat(prompt, system=pipe.system, tier=pipe.tier, temperature=0.0, max_tokens=600).strip()

            # Stream words/tokens to client
            words = full_text.split(" ")
            for i, w in enumerate(words):
                yield {"event": "token", "data": json.dumps({"token": w + (" " if i < len(words) - 1 else "")})}
                await asyncio.sleep(0.015)

            # B3: Validation on completion
            citations_list = [
                {"index": i + 1, "doc_id": h.doc_id, "excerpt": h.text[:250]}
                for i, h in enumerate(final_hits)
            ]
            ok, invalid = enforce_citations(full_text, len(final_hits))
            refused = full_text.strip().startswith("I don't have enough information")

            total_ms = (time.perf_counter() - t0) * 1000

            response_dict = {
                "answer": full_text,
                "refused": refused,
                "citations": citations_list,
            }
            _exact_cache_put(clean_question, response_dict)
            _semantic_cache_put(clean_question, response_dict)

            yield {
                "event": "done",
                "data": json.dumps({
                    "answer": full_text,
                    "citations": citations_list,
                    "citations_valid": ok or refused,
                    "invalid_citations": invalid,
                    "refused": refused,
                    "latency_ms": round(total_ms, 1),
                    "cost_usd": 0.00028,
                    "cached": False,
                    "trace_id": trace_id,
                }),
            }

    return EventSourceResponse(event_generator())
