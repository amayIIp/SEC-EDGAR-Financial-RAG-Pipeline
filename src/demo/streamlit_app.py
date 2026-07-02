from __future__ import annotations 
import os 
import time 
from typing import Any, Dict, List 
import altair as alt 
import pandas as pd 
import requests 
import streamlit as st 
st.set_page_config(
    page_title="SEC EDGAR Financial RAG Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.title("📑 SEC EDGAR Financial RAG Explorer")
st.markdown(
    "Query and audit multi-year company filings (10-K, 10-Q) using hybrid indexing "
    "retrieval and Cross-Encoder reranking."
)
st.sidebar.header("⚙️ Pipeline Configuration")
retrieval_mode = st.sidebar.selectbox(
    "Retrieval Mode",
    ["hybrid", "bm25_only", "vector_only"],
    index=0
)
embedding_prov = st.sidebar.selectbox("Embedding Model", ["openai", "bge"], index=0)
reranker_prov = st.sidebar.selectbox("Reranking Model", ["cohere", "bge", "none"], index=0)
generator_prov = st.sidebar.selectbox("LLM Generator", ["openai", "anthropic", "ollama"], index=0)
st.sidebar.divider()
st.sidebar.subheader("🎯 Metadata Filters")
filter_ticker = st.sidebar.text_input("Company Ticker Filter (e.g. AAPL)", value="AAPL")
filter_form = st.sidebar.selectbox("Filing Type", [None, "10-K", "10-Q", "8-K"], index=0)
filter_date_from = st.sidebar.date_input("Date From (Optional)", value=None)
filter_date_to = st.sidebar.date_input("Date To (Optional)", value=None)
filters_payload = {}
if filter_ticker.strip():
    filters_payload["ticker"] = filter_ticker.strip().upper()
if filter_form:
    filters_payload["filing_type"] = filter_form
if filter_date_from:
    filters_payload["date_from"] = str(filter_date_from)
if filter_date_to:
    filters_payload["date_to"] = str(filter_date_to)
col_left, col_right = st.columns([3, 2])
with col_left:
    st.subheader("❓ Pose your Question")
    user_query = st.text_input(
        "Ask a financial question about the filings:",
        value="What was Apple's total net revenue for fiscal year 2023?"
    )
    if st.button("Submit Query"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Executing pipeline (Search -> Fusion -> Rerank -> LLM)..."):
                try:
                    payload = {
                        "query": user_query,
                        "filters": filters_payload if filters_payload else None,
                        "mode": retrieval_mode,
                        "top_k": 15,
                        "top_n": 5,
                        "embedding_provider": embedding_prov,
                        "reranker_provider": reranker_prov,
                        "generator_provider": generator_prov
                    }
                    resp = requests.post(f"{API_URL}/debug", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    st.success("### Answer:")
                    st.write(data["answer"])
                    st.subheader("📚 Referenced Sources")
                    for idx, chunk_data in enumerate(data["reranked_hits"], start=1):
                        meta = chunk_data["chunk"]
                        header = (
                            f"Source [{idx}] | Ticker: {meta['ticker']} | Form: {meta['filing_type']} | "
                            f"Date: {meta['filing_date']} | Section: {meta['section']} (Score: {chunk_data['rerank_score']:.4f})"
                        )
                        with st.expander(header):
                            st.markdown(meta["text"])
                    st.divider()
                    st.subheader("🔍 Intermediate Search Diagnostics")
                    tab_bm25, tab_vec, tab_fused = st.tabs(["BM25 Hits", "Vector Hits", "RRF Fused Candidates"])
                    with tab_bm25:
                        for h in data["bm25_hits"][:5]:
                            st.caption(f"Score: {h['bm25_score']:.4f} | Chunk ID: {h['chunk']['chunk_id'][:12]}")
                            st.write(h["chunk"]["text"][:200] + "...")
                    with tab_vec:
                        for h in data["vector_hits"][:5]:
                            st.caption(f"Score: {h['vector_score']:.4f} | Chunk ID: {h['chunk']['chunk_id'][:12]}")
                            st.write(h["chunk"]["text"][:200] + "...")
                    with tab_fused:
                        for h in data["fused_hits"]:
                            st.caption(f"RRF Score: {h['rrf_score']:.6f} | BM25 Rank: {h['bm25_rank']} | Vector Rank: {h['vector_rank']}")
                            st.write(h["chunk"]["text"][:150] + "...")
                    st.session_state["latency_breakdown"] = data["latency_breakdown"]
                except Exception as exc:
                    st.error(f"Error querying API: {exc}")
with col_right:
    st.subheader("⚡ Latency breakdown (Observability)")
    if "latency_breakdown" in st.session_state:
        lat = st.session_state["latency_breakdown"]
        chart_data = pd.DataFrame([
            {"Stage": "Query Embedding", "Latency (ms)": lat.get("query_embedding", 0.0)},
            {"Stage": "Search & Fusion", "Latency (ms)": lat.get("hybrid_search", 0.0)},
            {"Stage": "Reranking", "Latency (ms)": lat.get("reranking", 0.0)},
            {"Stage": "Context Packing", "Latency (ms)": lat.get("context_packing", 0.0)},
            {"Stage": "LLM Generation", "Latency (ms)": lat.get("generation", 0.0)}
        ])
        chart = alt.Chart(chart_data).mark_bar(color="skyblue").encode(
            x=alt.X("Stage", sort=None),
            y="Latency (ms)"
        ).properties(width=300, height=250)
        st.altair_chart(chart, use_container_width=True)
        st.table(chart_data)
        st.write(f"⏱️ **Total Pipeline Latency:** {lat.get('total_pipeline', 0.0):.1f} ms")
    else:
        st.info("Run a query to see performance latency breakdowns.")
    st.divider()
    st.subheader("🤖 Live Search Modes Comparison")
    st.markdown("Compares latency and answers for BM25, Vector, and Hybrid modes side-by-side.")
    if st.button("Compare Modes Now"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            comparison_results = []
            modes_to_test = ["bm25_only", "vector_only", "hybrid"]
            with st.spinner("Querying all search modes..."):
                for m in modes_to_test:
                    try:
                        payload = {
                            "query": user_query,
                            "filters": filters_payload if filters_payload else None,
                            "mode": m,
                            "top_k": 15,
                            "top_n": 5,
                            "embedding_provider": embedding_prov,
                            "reranker_provider": reranker_prov,
                            "generator_provider": generator_prov
                        }
                        t_start = time.perf_counter()
                        resp = requests.post(f"{API_URL}/query", json=payload)
                        elapsed = (time.perf_counter() - t_start) * 1000
                        if resp.status_code == 200:
                            data = resp.json()
                            comparison_results.append({
                                "Mode": m,
                                "Latency (ms)": f"{elapsed:.1f} ms",
                                "Citations Count": len(data["citations"]),
                                "Answer Preview": data["answer"][:100] + "..."
                            })
                    except Exception as exc:
                        comparison_results.append({
                            "Mode": m,
                            "Latency (ms)": "Failed",
                            "Citations Count": 0,
                            "Answer Preview": str(exc)[:80]
                        })
            st.table(pd.DataFrame(comparison_results))
