from __future__ import annotations 
from typing import Any, Dict 
from fastapi import FastAPI, status, BackgroundTasks 
from fastapi.middleware.cors import CORSMiddleware 
from src.api.error_handlers import register_error_handlers 
from src.api.pipeline_runner import PipelineRunner 
from src.api.schemas import DebugResponse, QueryRequest, QueryResponse, IngestRequest 
from src.shared.config import cfg 
from src.shared.logging_setup import configure_logging 
configure_logging(cfg.profiling.enabled and "INFO" or "WARNING")
app = FastAPI(
    title="SEC RAG Pipeline API",
    description="Production-grade asynchronous RAG API server over SEC disclosures.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
register_error_handlers(app)
runner = PipelineRunner()
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    """
    Diagnostic route to verify that the API server is online and responding.
    """
    return {"status": "healthy", "service": "SEC RAG Pipeline"}
@app.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
def ingest_endpoint(request: IngestRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Schedules download, parsing, and indexing of filings in the background.
    """
    from src.ingestion.run_ingest import _resolve_cik
    cik = _resolve_cik(request.ticker)
    if not cik:
        raise ValueError(f"Ticker '{request.ticker}' is not registered or supported.")
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
@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Processes a financial question by executing retrieval, reranking, and generation.
    Returns a cited answer and a breakdown of stage latencies.
    """
    filters_dict = request.filters.model_dump() if request.filters else None
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
    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        latency_breakdown=result["latency_breakdown"],
        total_tokens=result["total_tokens"]
    )
@app.post("/debug", response_model=DebugResponse, status_code=status.HTTP_200_OK)
async def debug_endpoint(request: QueryRequest) -> DebugResponse:
    """
    Diagnostics route. Runs the query pipeline and exposes all intermediate state records
    (BM25 hits, vector hits, RRF rankings, reranker scores) alongside the final answer.
    """
    filters_dict = request.filters.model_dump() if request.filters else None
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
    return DebugResponse(
        answer=result["answer"],
        bm25_hits=result["raw_state"]["bm25_hits"],
        vector_hits=result["raw_state"]["vector_hits"],
        fused_hits=result["raw_state"]["fused_hits"],
        reranked_hits=result["raw_state"]["reranked_hits"],
        final_context=result["raw_state"]["final_context"],
        latency_breakdown=result["latency_breakdown"]
    )
