#!/usr/bin/env python3
"""Lab 7 — the observability dashboard, read from local traces.

    streamlit run labs/lab7/dashboard.py

`aip.tracing` writes one JSONL file per run to .aip_traces/. This page reads
them back. It is a teaching-scale stand-in for Langfuse / LangSmith / Phoenix;
the concept -- structured spans with a run id and a parent id -- is identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.config import settings

st.set_page_config(page_title="Aurora Assistant — Ops", layout="wide")
st.title("Aurora Policy Assistant — Operations & Observability")

runs = sorted(settings.trace_dir.glob("*.jsonl"), reverse=True)
if not runs:
    st.info(f"No traces yet in {settings.trace_dir}. Run some queries first.")
    st.stop()

chosen = st.sidebar.multiselect("runs", [p.stem for p in runs],
                                default=[runs[0].stem])
rows = [json.loads(l) for p in runs if p.stem in chosen
        for l in p.open(encoding="utf-8") if l.strip()]
if not rows:
    st.stop()

df = pd.DataFrame(rows)
df["ts"] = pd.to_datetime(df["ts"], unit="s")

c = st.columns(5)
c[0].metric("Spans", len(df))
total_cost = df.get("cost_usd", pd.Series([0], dtype=float)).fillna(0).sum()
c[1].metric("Total Cost", f"${total_cost:.4f}")
llm = df[df["name"] == "llm.call"]
c[2].metric("Model Calls", len(llm))
if len(llm):
    c[3].metric("Cache Hit Rate", f"{llm.get('cached', pd.Series([False])).fillna(False).mean():.0%}")
else:
    c[3].metric("Cache Hit Rate", "0%")
c[4].metric("Errors", int((df.get("status") == "error").sum()))

st.subheader("Latency by Stage (p50 / p95 Breakdown)")
# TODO C3: p50/p95 per span name. This is the table that answers
#          "which stage should I optimise?" -- see Lab 7 Part B4.
if "duration_ms" in df and len(df):
    stage = (df.groupby("name")["duration_ms"]
             .agg(n="count", p50="median",
                  p95=lambda s: s.quantile(0.95), total="sum")
             .sort_values("total", ascending=False))
    st.dataframe(stage, use_container_width=True)

st.subheader("Cost Over Time")
if "cost_usd" in df:
    cum = df.sort_values("ts").assign(cum=lambda d: d["cost_usd"].fillna(0).cumsum())
    st.line_chart(cum.set_index("ts")["cum"])

st.subheader("Errors & Guard Violations")
errs = df[df.get("status") == "error"]
if len(errs):
    st.dataframe(errs[["ts", "name", "error"]], use_container_width=True)
else:
    st.success("No error spans recorded in selected runs.")

# ---------------------------------------------------------------------------
# TODO C4: Alert Conditions and Operational Runbook
# ---------------------------------------------------------------------------
st.subheader("Active System Alerts & SLO Status (C4)")

# Alert 1: Refusal rate doubling (indicates broken/stale index)
ask_spans = df[df["name"].isin(["http.ask", "rag.answer"])]
refusal_spans = df[df.get("refused") == True]
refusal_rate = (len(refusal_spans) / max(1, len(ask_spans)))

# Alert 2: p95 latency above SLO threshold (6,000 ms)
http_spans = df[df["name"].str.startswith("http.ask", na=False)]
p95_latency = http_spans["duration_ms"].quantile(0.95) if (len(http_spans) and "duration_ms" in http_spans) else 0.0

col_a, col_b = st.columns(2)
with col_a:
    st.metric("Refusal Rate", f"{refusal_rate:.1%}", delta="SLO ceiling: 25.0%")
    if refusal_rate > 0.25:
        st.error("🚨 ALERT FIRED: Refusal rate doubled or breached 25% threshold!")
        st.markdown(
            "**Operational Runbook:**\n"
            "1. **Root cause:** A sudden surge in refusals almost always indicates index desynchronization or a missing/stale corpus file. RAG systems fail silently without raising HTTP 500s when no documents are matched.\n"
            "2. **Immediate Action:** Check the vector store index size via `/health`, verify `data/corpus/` file integrity, and re-warm embeddings if required."
        )
    else:
        st.success("Refusal rate is within acceptable operational baseline.")

with col_b:
    st.metric("p95 End-to-End Latency", f"{p95_latency:.0f} ms", delta="SLO: <= 6000 ms")
    if p95_latency > 6000:
        st.error("🚨 ALERT FIRED: p95 Latency violated 6,000 ms SLO budget!")
        st.markdown(
            "**Operational Runbook:**\n"
            "1. **Root cause:** Upstream provider latency spike, network throttling, or generation output token bloat.\n"
            "2. **Immediate Action:** Check `/metrics` for upstream retry counts, enforce max token limits, and enable caching."
        )
    else:
        st.success("p95 Latency conforms to production SLO.")
