# src/api/schemas.py
# This module defines the Pydantic schemas for our FastAPI endpoints.
# Pydantic validates incoming JSON request payloads against these schemas
# and guarantees that outgoing responses conform strictly to the defined formats.

from __future__ import annotations # Allow self-referencing type annotations.
from typing import Any, Dict, List, Optional # Type helpers.
from pydantic import BaseModel, Field # Pydantic classes for schema mapping and validation.

class QueryFilters(BaseModel):
    """
    Optional filter parameters to restrict search candidate scope.
    """
    ticker: Optional[str] = Field(default=None, description="Stock ticker symbol, e.g. AAPL")
    filing_type: Optional[str] = Field(default=None, description="Form type, e.g. 10-K or 10-Q")
    date_from: Optional[str] = Field(default=None, description="Earliest filing date as YYYY-MM-DD")
    date_to: Optional[str] = Field(default=None, description="Latest filing date as YYYY-MM-DD")

class IngestRequest(BaseModel):
    """
    Request body schema for POST /ingest.
    """
    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL")
    forms: List[str] = Field(default=["10-K", "10-Q"], description="SEC form types to download")
    limit: int = Field(default=3, description="Maximum number of filings to ingest")
    embedding_provider: str = Field(default="openai", description="Embedding model: 'openai' or 'bge'")

class QueryRequest(BaseModel):
    """
    Request body schema for POST /query and POST /debug.
    """
    query: str = Field(..., description="The natural language question to ask.")
    filters: Optional[QueryFilters] = Field(default=None, description="Optional metadata filters.")
    mode: str = Field(default="hybrid", description="Search mode: hybrid, bm25_only, or vector_only.")
    top_k: int = Field(default=15, description="Number of candidate chunks to fetch from indexes.")
    top_n: int = Field(default=5, description="Number of top reranked chunks to feed as context.")
    embedding_provider: Optional[str] = Field(default=None, description="Override embedding model provider.")
    reranker_provider: Optional[str] = Field(default=None, description="Override reranker model provider.")
    generator_provider: Optional[str] = Field(default=None, description="Override LLM generator provider.")
    generator_model: Optional[str] = Field(default=None, description="Override specific LLM model name.")

class Citation(BaseModel):
    """
    Individual citation record returned in /query answers.
    """
    chunk_id: str = Field(..., description="Unique deterministic identifier of the cited chunk.")
    citation_tag: str = Field(..., description="Label tag matched in the answer text, e.g. [1]")
    text_snippet: str = Field(..., description="Text text fragment matching the cited chunk.")
    ticker: str = Field(..., description="Source company ticker.")
    form: str = Field(..., description="Source form type.")
    filing_date: str = Field(..., description="Source filing date.")
    section: str = Field(..., description="Source filing section code, e.g. Item 7.")

class QueryResponse(BaseModel):
    """
    Unified JSON response schema for POST /query.
    """
    answer: str = Field(..., description="Synthesized cited answer from the LLM.")
    citations: List[Citation] = Field(..., description="Citations mapping cited tags to source text and metadata.")
    latency_breakdown: Dict[str, float] = Field(..., description="Latency (ms) of each pipeline stage.")
    total_tokens: int = Field(..., description="Total tokens consumed in generation (prompt + completion).")

class DebugResponse(BaseModel):
    """
    Verbose diagnostic response schema for POST /debug.
    """
    answer: str = Field(..., description="Synthesized cited answer.")
    bm25_hits: List[Dict[str, Any]] = Field(..., description="Raw hits from OpenSearch index.")
    vector_hits: List[Dict[str, Any]] = Field(..., description="Raw hits from Qdrant vector index.")
    fused_hits: List[Dict[str, Any]] = Field(..., description="Top RRF-fused hits before reranking.")
    reranked_hits: List[Dict[str, Any]] = Field(..., description="Top hits after reranking.")
    final_context: List[Dict[str, Any]] = Field(..., description="Text blocks packed into prompt context.")
    latency_breakdown: Dict[str, float] = Field(..., description="Latency (ms) of each stage.")
