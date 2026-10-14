#!/usr/bin/env python3
"""Lab 7 — Streamlit front end.

    streamlit run labs/lab7/ui.py

Requires the service to be running:
    uvicorn labs.lab7.service:app --port 8000

The one non-negotiable UI requirement: **citations must be expandable to show
the source text.** Grounding the user cannot check is decoration.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
REVIEW_QUEUE = ROOT / "data/review_queue.jsonl"

st.set_page_config(page_title="Aurora Policy Assistant", layout="centered")

API = st.sidebar.text_input("Service URL", "http://localhost:8000")
stream_mode = st.sidebar.checkbox("Enable streaming (/ask/stream)", value=False)
mode = st.sidebar.selectbox("Mode", ["rag", "tools"])

st.title("Aurora Policy Assistant")
st.caption("Answers come only from Aurora's policy documents. "
           "Every claim is cited. When the documents do not cover a question, "
           "the assistant says so instead of guessing.")

q = st.text_input("Ask a question",
                  placeholder="How long do I have to file a reimbursement claim?")

if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if st.button("Ask", type="primary") and q:
    st.session_state.last_question = q
    if stream_mode:
        answer_placeholder = st.empty()
        full_text = ""
        citations_data = []
        meta_data = {}

        try:
            with requests.post(
                f"{API}/ask/stream",
                json={"question": q, "mode": mode},
                stream=True,
                timeout=60,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        payload_str = line_str[6:].strip()
                        try:
                            payload = json.loads(payload_str)
                            if "token" in payload:
                                full_text += payload["token"]
                                answer_placeholder.markdown(full_text + "▌")
                            elif "citations" in payload:
                                meta_data = payload
                                citations_data = payload.get("citations", [])
                        except Exception:  # noqa: BLE001, S110
                            pass
            answer_placeholder.markdown(full_text)
            data = {
                "answer": full_text,
                "refused": meta_data.get("refused", False),
                "citations": citations_data,
                "latency_ms": meta_data.get("latency_ms", 0),
                "cost_usd": meta_data.get("cost_usd", 0),
                "cached": meta_data.get("cached", False),
                "trace_id": meta_data.get("trace_id", ""),
            }
            st.session_state.last_response = data
        except Exception as exc:  # noqa: BLE001
            st.error(f"Streaming error: {exc}")
            st.stop()
    else:
        with st.spinner("Thinking..."):
            try:
                r = requests.post(f"{API}/ask", json={"question": q, "mode": mode}, timeout=60)
                r.raise_for_status()
                data = r.json()
                st.session_state.last_response = data
            except requests.HTTPError as exc:
                st.error(f"{exc.response.status_code}: {exc.response.text[:300]}")
                st.stop()
            except requests.RequestException as exc:
                st.error(f"Service unreachable: {exc}")
                st.stop()

        if data.get("refused"):
            st.warning(data["answer"])
        else:
            st.markdown(data["answer"])

    # TODO A4: render citations as expanders showing the source excerpt.
    #          A citation the user cannot open is not grounding.
    data = st.session_state.last_response
    if data and data.get("citations"):
        st.subheader("Sources & Citations")
        for c in data.get("citations", []):
            with st.expander(f"[{c['index']}] {c['doc_id']}"):
                st.markdown(f"**Excerpt:**\n> {c['excerpt']}")

    if data:
        cols = st.columns(4)
        cols[0].metric("Latency", f"{data.get('latency_ms', 0):.0f} ms")
        cols[1].metric("Cost", f"${data.get('cost_usd', 0):.5f}")
        cols[2].metric("Cached", "Yes" if data.get("cached") else "No")
        cols[3].metric("Sources Cited", len(data.get("citations", [])))
        st.caption(f"Trace ID: `{data.get('trace_id', '')}`")

        # TODO stretch: a thumbs-down button that appends the case to a review queue.
        # That queue is how real golden sets get built.
        st.markdown("---")
        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])
        with fb_col1:
            if st.button("👍 Good Answer"):
                st.success("Feedback recorded: Helpful.")
        with fb_col2:
            if st.button("👎 Flag Issue"):
                REVIEW_QUEUE.parent.mkdir(parents=True, exist_ok=True)
                review_entry = {
                    "question": st.session_state.last_question,
                    "answer": data["answer"],
                    "citations": data.get("citations", []),
                    "trace_id": data.get("trace_id", ""),
                    "timestamp": time.time(),
                }
                with REVIEW_QUEUE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(review_entry, ensure_ascii=False) + "\n")
                st.warning("Case submitted to review queue for evaluation set expansion.")
