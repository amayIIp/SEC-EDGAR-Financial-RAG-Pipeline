# src/demo/streamlit_app.py
# This module implements the Streamlit dashboard user interface.
# It communicates with the FastAPI server running on port 8000 via REST calls.
#
# =========================================================================================
# Advanced Concept: RAG Observability Dashboard
# To evaluate and demo RAG, a simple chatbot text input is not enough.
# We build an interactive workspace displaying:
# 1. Answer & Citations: Collapsible text boxes showing matching context snippets.
# 2. Latency Profiling: A bar chart detailing how many milliseconds each stage took.
# 3. Intermediate State Inspection: Tabs exposing raw BM25 hits and vector similarity points.
# 4. Mode Comparison: A live side-by-side speed and score table comparing BM25 vs Vector vs Hybrid.
# =========================================================================================

from __future__ import annotations # Allow self-referencing type annotations.
import os # Standard library module to read environmental properties.
import time # Standard library module to measure loading time.
from typing import Any, Dict, List # Type helpers.
import altair as alt # Declarative statistical visualization library.
import pandas as pd # Data manipulation library.
import requests # HTTP client library.
import streamlit as st # Streamlit web server.

# Set Streamlit page settings.
st.set_page_config(
    page_title="SEC EDGAR Financial RAG Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API service base URL, loading from environment (useful in Docker Compose bridge networks).
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("📑 SEC EDGAR Financial RAG Explorer")
st.markdown(
    "Query and audit multi-year company filings (10-K, 10-Q) using hybrid indexing "
    "retrieval and Cross-Encoder reranking."
)

# --- Sidebar Parameter Selection ---
st.sidebar.header("⚙️ Pipeline Configuration")

# Retrieval Mode selection.
retrieval_mode = st.sidebar.selectbox(
    "Retrieval Mode",
    ["hybrid", "bm25_only", "vector_only"],
    index=0
)

# Model configuration overrides.
embedding_prov = st.sidebar.selectbox("Embedding Model", ["openai", "bge"], index=0)
reranker_prov = st.sidebar.selectbox("Reranking Model", ["cohere", "bge", "none"], index=0)
generator_prov = st.sidebar.selectbox("LLM Generator", ["openai", "anthropic", "ollama"], index=0)

st.sidebar.divider()

# Optional metadata filters.
st.sidebar.subheader("🎯 Metadata Filters")
filter_ticker = st.sidebar.text_input("Company Ticker Filter (e.g. AAPL)", value="AAPL")
filter_form = st.sidebar.selectbox("Filing Type", [None, "10-K", "10-Q", "8-K"], index=0)
filter_date_from = st.sidebar.date_input("Date From (Optional)", value=None)
filter_date_to = st.sidebar.date_input("Date To (Optional)", value=None)

# Build filters dictionary.
filters_payload = {}
if filter_ticker.strip():
    filters_payload["ticker"] = filter_ticker.strip().upper()
if filter_form:
    filters_payload["filing_type"] = filter_form
if filter_date_from:
    filters_payload["date_from"] = str(filter_date_from)
if filter_date_to:
    filters_payload["date_to"] = str(filter_date_to)

# Layout division.
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("❓ Pose your Question")
    # Text input for query.
    user_query = st.text_input(
        "Ask a financial question about the filings:",
        value="What was Apple's total net revenue for fiscal year 2023?"
    )
    
    # Run query.
    if st.button("Submit Query"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Executing pipeline (Search -> Fusion -> Rerank -> LLM)..."):
                try:
                    # Construct API payload.
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
                    
                    # Issue request.
                    resp = requests.post(f"{API_URL}/debug", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    
                    # Display response answer.
                    st.success("### Answer:")
                    st.write(data["answer"])
                    
                    # Reranked Context sources.
                    st.subheader("📚 Referenced Sources")
                    for idx, chunk_data in enumerate(data["reranked_hits"], start=1):
                        meta = chunk_data["chunk"]
                        header = (
                            f"Source [{idx}] | Ticker: {meta['ticker']} | Form: {meta['filing_type']} | "
                            f"Date: {meta['filing_date']} | Section: {meta['section']} (Score: {chunk_data['rerank_score']:.4f})"
                        )
                        with st.expander(header):
                            st.markdown(meta["text"])
                            
                    # intermediate debug panel.
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

                    # Cache latency for the right-hand panel.
                    st.session_state["latency_breakdown"] = data["latency_breakdown"]
                    
                except Exception as exc:
                    st.error(f"Error querying API: {exc}")

with col_right:
    st.subheader("⚡ Latency breakdown (Observability)")
    
    # If latency info exists, render Altair bar chart.
    if "latency_breakdown" in st.session_state:
        lat = st.session_state["latency_breakdown"]
        
        # Convert dictionary to Pandas DataFrame.
        chart_data = pd.DataFrame([
            {"Stage": "Query Embedding", "Latency (ms)": lat.get("query_embedding", 0.0)},
            {"Stage": "Search & Fusion", "Latency (ms)": lat.get("hybrid_search", 0.0)},
            {"Stage": "Reranking", "Latency (ms)": lat.get("reranking", 0.0)},
            {"Stage": "Context Packing", "Latency (ms)": lat.get("context_packing", 0.0)},
            {"Stage": "LLM Generation", "Latency (ms)": lat.get("generation", 0.0)}
        ])
        
        # Build Altair chart.
        chart = alt.Chart(chart_data).mark_bar(color="skyblue").encode(
            x=alt.X("Stage", sort=None),
            y="Latency (ms)"
        ).properties(width=300, height=250)
        
        st.altair_chart(chart, use_container_width=True)
        
        # Print total time.
        st.table(chart_data)
        st.write(f"⏱️ **Total Pipeline Latency:** {lat.get('total_pipeline', 0.0):.1f} ms")
    else:
        st.info("Run a query to see performance latency breakdowns.")
        
    st.divider()
    
    # ── Live Search Modes Comparison ──
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
                        # Construct API payload.
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
                        
            # Render comparison grid.
            st.table(pd.DataFrame(comparison_results))
