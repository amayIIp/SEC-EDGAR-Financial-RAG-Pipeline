from __future__ import annotations  
import hashlib  
import uuid     
from datetime import datetime  
from enum import Enum          
from typing import Any, Optional  
from pydantic import BaseModel, Field, field_validator  
class FilingType(str, Enum):
    """Valid SEC form types this pipeline handles."""
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    EIGHT_K = "8-K"
class ChunkingStrategy(str, Enum):
    """The three chunking approaches we compare in Phase 2."""
    NAIVE_FIXED = "naive_fixed"
    STRUCTURE_AWARE = "structure_aware"
    SEMANTIC = "semantic"
class EmbeddingProvider(str, Enum):
    """Which embedding model was used to generate the vectors in a collection."""
    OPENAI = "openai"
    BGE = "bge"
class RetrievalMode(str, Enum):
    """Retrieval pathway used for a query."""
    BM25_ONLY = "bm25_only"
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"
class RerankerProvider(str, Enum):
    """Which reranker was applied after retrieval fusion."""
    COHERE = "cohere"
    BGE = "bge"
    NONE = "none"
class GenerationProvider(str, Enum):
    """Which LLM backend produces the final answer."""
    OPENAI = "openai"          
    ANTHROPIC = "anthropic"    
    OLLAMA = "ollama"          
class FilingMetadata(BaseModel):
    """
    Identifies a single SEC filing precisely.
    Attached to every Chunk and SearchResult so any downstream stage can filter
    or cite by company, form type, and date without re-fetching the filing.
    """
    ticker: str
    cik: str
    filing_type: FilingType
    filing_date: str
    accession_number: str
    fiscal_year: int
    local_path: Optional[str] = None
    @field_validator("cik")
    @classmethod
    def pad_cik(cls, v: str) -> str:
        return v.zfill(10)
class ParsedSection(BaseModel):
    """
    One logical section of a parsed SEC filing (e.g. 'Item 1A. Risk Factors').
    Produced by src/parsing/html_parser.py; consumed by src/chunking/*.
    """
    item_id: str
    title: str
    content: str
    tables: list[dict[str, Any]] = Field(default_factory=list)
    start_char_offset: Optional[int] = None
class ParsedFiling(BaseModel):
    """
    Complete output of the HTML parser for one SEC filing document.
    Written to data/parsed/{ticker}/{form}_{date}.json.
    """
    metadata: FilingMetadata
    sections: list[ParsedSection]
    total_content_chars: int = 0
    had_parse_warnings: bool = False
class Chunk(BaseModel):
    """
    A single indexable unit of text produced by a chunker.
    This is the primary object that flows into OpenSearch and Qdrant.
    Every field here is either:
      (a) stored as a payload field in Qdrant for filtered vector search, OR
      (b) mapped to an index field in OpenSearch for BM25 + keyword filtering.
    """
    chunk_id: str
    text: str
    strategy: ChunkingStrategy
    chunk_index: int
    is_table: bool = False
    ticker: str
    cik: str
    filing_type: FilingType
    filing_date: str
    fiscal_year: int
    accession_number: str
    section: str
    section_title: str
    token_count: int = 0
    @staticmethod
    def make_id(ticker: str, accession_number: str, strategy: str, chunk_index: int) -> str:
        """
        Generates a deterministic, collision-resistant chunk ID.
        SHA-256 produces a 64-character hex string that is unique per
        (filing, strategy, position) triple without needing a central counter.
        """
        raw = f"{ticker}::{accession_number}::{strategy}::{chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()
class BM25Result(BaseModel):
    """
    One hit returned by an OpenSearch BM25 query.
    Wraps the raw Chunk payload with the BM25 relevance score and rank.
    """
    chunk: Chunk
    bm25_score: float
    bm25_rank: int
class VectorResult(BaseModel):
    """
    One hit returned by a Qdrant vector similarity search.
    Wraps the raw Chunk payload with the cosine similarity score and rank.
    """
    chunk: Chunk
    vector_score: float
    vector_rank: int
class FusedResult(BaseModel):
    """
    One chunk after Reciprocal Rank Fusion has combined BM25 and vector ranks.
    Carries both original ranks for debugging and the new fused score.
    """
    chunk: Chunk
    rrf_score: float
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None
class RankedResult(BaseModel):
    """
    One chunk after the cross-encoder reranker has re-scored the fused candidates.
    This is the final ranked object passed to the generation stage.
    """
    chunk: Chunk
    rerank_score: float
    final_rank: int
    rrf_score: Optional[float] = None
class CitedChunk(BaseModel):
    """
    A chunk that was included in the context window and the citation tag
    the prompt template assigned to it (e.g. '[3]').
    """
    ranked_result: RankedResult
    citation_tag: str
    packed_text: str
    packed_token_count: int
class GenerationOutput(BaseModel):
    """
    Complete output from the generation stage for one user query.
    Returned by src/api/pipeline_runner.py and serialised to the /query response.
    """
    query: str
    answer: str
    cited_tags: list[str]
    context_chunks: list[CitedChunk]
    context_utilisation_ratio: float
    provider: GenerationProvider
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
class StageProfile(BaseModel):
    """
    One timing record written by the @profile_stage decorator.
    Each pipeline run produces many of these, one per decorated function call.
    """
    run_id: str
    stage_name: str
    started_at: str
    duration_ms: float
    extra: dict[str, Any] = Field(default_factory=dict)
    @staticmethod
    def now_iso() -> str:
        """Returns the current UTC time as an ISO-8601 string."""
        return datetime.utcnow().isoformat() + "Z"
class QueryType(str, Enum):
    """Classification of an eval query by the reasoning skill it tests."""
    FACTUAL = "factual"          
    COMPARATIVE = "comparative"  
    QUALITATIVE = "qualitative"  
class EvalQuery(BaseModel):
    """
    One entry in the 50-query golden evaluation set (data/eval/eval_set.jsonl).
    Phase 7 reads this file and runs the full pipeline for each entry.
    """
    id: str
    query: str
    query_type: QueryType
    ticker: Optional[str] = None
    filing_type: Optional[FilingType] = None
    fiscal_year: Optional[int] = None
    target_section: Optional[str] = None
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    expected_answer: Optional[str] = None
    notes: Optional[str] = None
class EvalResult(BaseModel):
    """
    Metrics produced for one EvalQuery after running the full pipeline.
    Collected by run_eval_matrix.py into the final comparison CSV.
    """
    query_id: str
    chunking_strategy: ChunkingStrategy
    retrieval_mode: RetrievalMode
    reranker: RerankerProvider
    embedding_provider: EmbeddingProvider
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    precision_at_5: float
    precision_at_10: float
    mrr: float   
    faithfulness: Optional[float] = None        
    answer_relevancy: Optional[float] = None    
    context_precision: Optional[float] = None   
    context_recall: Optional[float] = None      
    llm_judge_score: Optional[float] = None     
    latency_embedding_ms: float = 0.0
    latency_bm25_ms: float = 0.0
    latency_vector_ms: float = 0.0
    latency_rrf_ms: float = 0.0
    latency_rerank_ms: float = 0.0
    latency_generation_ms: float = 0.0
    latency_total_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_utilisation_ratio: float = 0.0
