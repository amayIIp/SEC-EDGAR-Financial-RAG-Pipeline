# src/api/main.py
# This module is the entry-point factory for the FastAPI web server.
# It initializes the FastAPI app, registers middleware, configures exception routing,
# and exposes HTTP endpoints: /query, /debug, and /health.

from __future__ import annotations # Allow self-referencing type annotations.
from typing import Any, Dict # Type helpers.
from fastapi import FastAPI, status, BackgroundTasks # FastAPI classes.
from fastapi.middleware.cors import CORSMiddleware # CORS middleware.
from src.api.error_handlers import register_error_handlers # Custom exception handler registration.
from src.api.pipeline_runner import PipelineRunner # Pipeline coordinator.
from src.api.schemas import DebugResponse, QueryRequest, QueryResponse, IngestRequest # Pydantic schemas.
from src.shared.config import cfg # Config loader singleton.
from src.shared.logging_setup import configure_logging # Logging setup.

# Configure structured logging.
configure_logging(cfg.profiling.enabled and "INFO" or "WARNING")

# Initialize the FastAPI app instance.
app = FastAPI(
    title="SEC RAG Pipeline API",
    description="Production-grade asynchronous RAG API server over SEC disclosures.",
    version="1.0.0"
)

# Apply CORS (Cross-Origin Resource Sharing) middleware.
# This permits requests originating from frontend frameworks (like Streamlit on port 8501)
# to query the backend server on port 8000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register the global exception middleware handlers.
register_error_handlers(app)

# Initialize the PipelineRunner singleton.
runner = PipelineRunner()

# ── Endpoint 1: Healthcheck ──
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    """
    Diagnostic route to verify that the API server is online and responding.
    """
    return {"status": "healthy", "service": "SEC RAG Pipeline"}

# ── Endpoint 1b: Background Ingestion ──
@app.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
def ingest_endpoint(request: IngestRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Schedules download, parsing, and indexing of filings in the background.
    """
    # Enforce CIK ticker mapping exists.
    from src.ingestion.run_ingest import _resolve_cik
    cik = _resolve_cik(request.ticker)
    if not cik:
        raise ValueError(f"Ticker '{request.ticker}' is not registered or supported.")
        
    # Queue execution.
    background_tasks.add_task(
        runner.ingest,
        ticker=request.ticker,
        forms=request.forms,
        limit=request.limit,
        embedding_provider=request.embedding_provider
    )
    
    return {
        "status": "queued",
        "message": f"Ingestion process for {request.ticker.upper()} has been scheduled in the background."
    }

# ── Endpoint 2: QA Search Query ──
@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Processes a financial question by executing retrieval, reranking, and generation.
    Returns a cited answer and a breakdown of stage latencies.
    """
    # Map filters Pydantic model to raw dict if provided.
    filters_dict = request.filters.model_dump() if request.filters else None
    
    # Run the orchestrator pipeline.
    result = await runner.run_pipeline(
        query=request.query,
        filters=filters_dict,
        mode=request.mode,
        top_k=request.top_k,
        top_n=request.top_n,
        embedding_provider=request.embedding_provider,
        reranker_provider=request.reranker_provider,
        generator_provider=request.generator_provider,
        generator_model=request.generator_model
    )
    
    # Return formatted response matching QueryResponse schema.
    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        latency_breakdown=result["latency_breakdown"],
        total_tokens=result["total_tokens"]
    )

# ── Endpoint 3: Diagnostics Debugging ──
@app.post("/debug", response_model=DebugResponse, status_code=status.HTTP_200_OK)
async def debug_endpoint(request: QueryRequest) -> DebugResponse:
    """
    Diagnostics route. Runs the query pipeline and exposes all intermediate state records
    (BM25 hits, vector hits, RRF rankings, reranker scores) alongside the final answer.
    """
    filters_dict = request.filters.model_dump() if request.filters else None
    
    # Execute query.
    result = await runner.run_pipeline(
        query=request.query,
        filters=filters_dict,
        mode=request.mode,
        top_k=request.top_k,
        top_n=request.top_n,
        embedding_provider=request.embedding_provider,
        reranker_provider=request.reranker_provider,
        generator_provider=request.generator_provider,
        generator_model=request.generator_model
    )
    
    # Return verbose response.
    return DebugResponse(
        answer=result["answer"],
        bm25_hits=result["raw_state"]["bm25_hits"],
        vector_hits=result["raw_state"]["vector_hits"],
        fused_hits=result["raw_state"]["fused_hits"],
        reranked_hits=result["raw_state"]["reranked_hits"],
        final_context=result["raw_state"]["final_context"],
        latency_breakdown=result["latency_breakdown"]
    )
