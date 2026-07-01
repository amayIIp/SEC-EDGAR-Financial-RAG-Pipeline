# src/shared/models.py
#
# Shared dataclasses used across every pipeline stage.
# Defining them once here prevents the "drift problem" where ingestion calls a
# field "filing_date" but retrieval calls it "date" and the API calls it
# "submission_date" — three names for the same thing, discovered at 2 AM.
#
# DESIGN RULES FOR THIS FILE:
#   1. Only pure data containers here — no business logic, no I/O, no imports
#      from other src.* modules (that would create circular dependencies).
#   2. Use Pydantic BaseModel (not @dataclass) so every model gets automatic
#      JSON serialisation, field validation, and clear error messages for free.
#   3. Every field has an explicit type annotation and a docstring-style comment.
#      Optional fields carry a default so callers don't have to supply them.

from __future__ import annotations  # allows forward references in type hints

import hashlib  # standard-library module: generates deterministic chunk IDs via SHA-256
import uuid     # standard-library module: used to create UUID4 run identifiers
from datetime import datetime  # standard-library: timestamps on profiling events
from enum import Enum          # standard-library: defines fixed sets of valid string values
from typing import Any, Optional  # type-hint helpers

from pydantic import BaseModel, Field, field_validator  # Pydantic v2 model machinery


# =============================================================================
# Enumerations — finite sets of valid string values
# Using Enum means a typo like "hybrd" is caught at parse time, not silently
# passed to a database and returning zero results with no error message.
# =============================================================================

class FilingType(str, Enum):
    """Valid SEC form types this pipeline handles."""
    # Annual report — the most comprehensive filing, published once per fiscal year.
    TEN_K = "10-K"
    # Quarterly report — abbreviated, published three times per fiscal year.
    TEN_Q = "10-Q"
    # Current report — material events (earnings releases, executive changes, M&A).
    EIGHT_K = "8-K"


class ChunkingStrategy(str, Enum):
    """The three chunking approaches we compare in Phase 2."""
    # Fixed-size baseline: 512 tokens, 15% overlap, ignores section boundaries.
    NAIVE_FIXED = "naive_fixed"
    # Splits on SEC Item headers first, then recursively on paragraphs/sentences.
    STRUCTURE_AWARE = "structure_aware"
    # Cosine-similarity boundary detection over sentence embeddings.
    SEMANTIC = "semantic"


class EmbeddingProvider(str, Enum):
    """Which embedding model was used to generate the vectors in a collection."""
    # Cloud API: text-embedding-3-small, 1536 dimensions.
    OPENAI = "openai"
    # Self-hosted: BAAI/bge-large-en-v1.5, 1024 dimensions.
    BGE = "bge"


class RetrievalMode(str, Enum):
    """Retrieval pathway used for a query."""
    # BM25 keyword search only (OpenSearch).
    BM25_ONLY = "bm25_only"
    # Dense vector search only (Qdrant).
    VECTOR_ONLY = "vector_only"
    # Both combined via Reciprocal Rank Fusion.
    HYBRID = "hybrid"


class RerankerProvider(str, Enum):
    """Which reranker was applied after retrieval fusion."""
    # Cohere's cloud-hosted rerank-english-v3.0 cross-encoder.
    COHERE = "cohere"
    # Self-hosted BAAI/bge-reranker-v2-m3 CrossEncoder.
    BGE = "bge"
    # No reranking — RRF-fused order is passed directly to generation.
    NONE = "none"


class GenerationProvider(str, Enum):
    """Which LLM backend produces the final answer."""
    OPENAI = "openai"          # gpt-4o-mini (or any OpenAI chat model)
    ANTHROPIC = "anthropic"    # Claude 3 Haiku / Sonnet
    OLLAMA = "ollama"          # Local Llama 3.1 8B via Ollama REST API


# =============================================================================
# Core domain models — data that flows between pipeline stages
# =============================================================================

class FilingMetadata(BaseModel):
    """
    Identifies a single SEC filing precisely.
    Attached to every Chunk and SearchResult so any downstream stage can filter
    or cite by company, form type, and date without re-fetching the filing.
    """

    # Stock exchange ticker symbol, e.g. "AAPL".
    ticker: str

    # 10-digit zero-padded SEC Central Index Key that uniquely identifies the filer.
    # Required because ticker symbols can be reused after delistings; CIK never changes.
    cik: str

    # Type of SEC form (10-K, 10-Q, or 8-K).
    filing_type: FilingType

    # Filing date as reported by SEC EDGAR, in ISO-8601 format "YYYY-MM-DD".
    filing_date: str

    # SEC accession number that uniquely identifies this specific submission.
    # Format: XXXXXXXXXX-YY-ZZZZZZ (issuer CIK + year + sequence).
    accession_number: str

    # Fiscal year the filing covers (e.g. 2023).  Derived from filing_date by
    # ingestion; stored here so we can filter "FY2023 only" without date math.
    fiscal_year: int

    # Absolute local file path to the raw HTML file on disk.
    # Used by parser and chunker to open the file; not exposed in API responses.
    local_path: Optional[str] = None

    @field_validator("cik")
    @classmethod
    def pad_cik(cls, v: str) -> str:
        # SEC CIKs are always 10 digits — left-pad with zeros if shorter.
        # This ensures consistent storage and filtering across all modules.
        return v.zfill(10)


class ParsedSection(BaseModel):
    """
    One logical section of a parsed SEC filing (e.g. 'Item 1A. Risk Factors').
    Produced by src/parsing/html_parser.py; consumed by src/chunking/*.
    """

    # Normalised section identifier, e.g. "Item 1A" or "Item 7".
    item_id: str

    # Human-readable section title extracted from the heading, e.g. "Risk Factors".
    title: str

    # Full plain-text content of this section after stripping HTML tags and boilerplate.
    content: str

    # Structured tables found within this section.
    # Each table dict has keys: caption (str), rows (list[list[str]]), raw_text (str).
    tables: list[dict[str, Any]] = Field(default_factory=list)

    # Character offset where this section starts in the original HTML document.
    # Useful for debugging mis-identified section boundaries.
    start_char_offset: Optional[int] = None


class ParsedFiling(BaseModel):
    """
    Complete output of the HTML parser for one SEC filing document.
    Written to data/parsed/{ticker}/{form}_{date}.json.
    """

    # Metadata that identifies this filing.
    metadata: FilingMetadata

    # Ordered list of extracted sections in document order.
    sections: list[ParsedSection]

    # Total character length of all section content combined.
    # Computed at parse time; used by the chunking stats script.
    total_content_chars: int = 0

    # True if the parser encountered recoverable errors (logged but not fatal).
    had_parse_warnings: bool = False


class Chunk(BaseModel):
    """
    A single indexable unit of text produced by a chunker.
    This is the primary object that flows into OpenSearch and Qdrant.
    Every field here is either:
      (a) stored as a payload field in Qdrant for filtered vector search, OR
      (b) mapped to an index field in OpenSearch for BM25 + keyword filtering.
    """

    # Deterministic identifier: SHA-256 hex of (ticker + accession + strategy + index).
    # Deterministic means re-running the chunker on the same input always produces
    # the same IDs, so eval set 'relevant_chunk_ids' references remain valid.
    chunk_id: str

    # The actual text content of this chunk, ready for embedding and BM25 indexing.
    text: str

    # Which chunking strategy produced this chunk.
    strategy: ChunkingStrategy

    # Sequential position of this chunk within its source document.
    # Allows the context packer to re-order chunks into document reading order.
    chunk_index: int

    # True if this chunk came from an HTML table (not a paragraph).
    # Table chunks are handled differently in the context packer.
    is_table: bool = False

    # Filing identification and classification metadata (all filterable in both DBs).
    ticker: str
    cik: str
    filing_type: FilingType
    filing_date: str
    fiscal_year: int
    accession_number: str

    # SEC section this chunk belongs to, e.g. "Item 1A" or "Item 7".
    section: str

    # Human-readable section title, e.g. "Risk Factors".
    section_title: str

    # Token count of this chunk (using tiktoken cl100k_base).
    # Used to enforce context-window limits during generation.
    token_count: int = 0

    @staticmethod
    def make_id(ticker: str, accession_number: str, strategy: str, chunk_index: int) -> str:
        """
        Generates a deterministic, collision-resistant chunk ID.
        SHA-256 produces a 64-character hex string that is unique per
        (filing, strategy, position) triple without needing a central counter.
        """
        # Concatenate the four components that uniquely identify this chunk.
        raw = f"{ticker}::{accession_number}::{strategy}::{chunk_index}"
        # Hash the concatenated string; take the full 64-char hex digest.
        return hashlib.sha256(raw.encode()).hexdigest()


# =============================================================================
# Retrieval result models — output of OpenSearch, Qdrant, and fusion
# =============================================================================

class BM25Result(BaseModel):
    """
    One hit returned by an OpenSearch BM25 query.
    Wraps the raw Chunk payload with the BM25 relevance score and rank.
    """

    # The underlying chunk — all metadata and text are preserved.
    chunk: Chunk

    # BM25 relevance score assigned by OpenSearch (higher = more relevant).
    # Cannot be compared directly to vector_score — different scales entirely.
    bm25_score: float

    # 1-based rank position in the BM25 result list (1 = top hit).
    bm25_rank: int


class VectorResult(BaseModel):
    """
    One hit returned by a Qdrant vector similarity search.
    Wraps the raw Chunk payload with the cosine similarity score and rank.
    """

    chunk: Chunk

    # Cosine similarity between the query vector and this chunk's stored vector.
    # Range: -1.0 to 1.0 for cosine; typically 0.6–0.95 for relevant SEC text.
    vector_score: float

    # 1-based rank position in the vector result list (1 = most similar).
    vector_rank: int


class FusedResult(BaseModel):
    """
    One chunk after Reciprocal Rank Fusion has combined BM25 and vector ranks.
    Carries both original ranks for debugging and the new fused score.
    """

    chunk: Chunk

    # RRF score = Σ 1/(k + rank_i).  Higher is better.
    rrf_score: float

    # Original ranks from each system. None if the chunk was not returned by
    # that system (it appeared only in the other system's result list).
    bm25_rank: Optional[int] = None
    vector_rank: Optional[int] = None


class RankedResult(BaseModel):
    """
    One chunk after the cross-encoder reranker has re-scored the fused candidates.
    This is the final ranked object passed to the generation stage.
    """

    chunk: Chunk

    # Score assigned by the reranker (Cohere or BGE CrossEncoder).
    # Higher = more relevant to the query.  Not comparable across providers.
    rerank_score: float

    # 1-based position in the final reranked list (1 = most relevant).
    final_rank: int

    # Pre-rerank RRF score — kept for A/B comparison in the profiling phase.
    rrf_score: Optional[float] = None


# =============================================================================
# Generation models — output of the context packing and LLM stages
# =============================================================================

class CitedChunk(BaseModel):
    """
    A chunk that was included in the context window and the citation tag
    the prompt template assigned to it (e.g. '[3]').
    """

    # The ranked chunk being cited.
    ranked_result: RankedResult

    # Citation bracket label used in the prompt and answer, e.g. "[1]".
    citation_tag: str

    # Text actually sent to the LLM (may be compressed if compression is enabled).
    packed_text: str

    # Token count of packed_text.
    packed_token_count: int


class GenerationOutput(BaseModel):
    """
    Complete output from the generation stage for one user query.
    Returned by src/api/pipeline_runner.py and serialised to the /query response.
    """

    # The user's original query string.
    query: str

    # The LLM-generated answer text.
    answer: str

    # Which citation tags (e.g. "[1]", "[3]") actually appear in the answer text.
    # Populated by citation_checker.py after generation.
    cited_tags: list[str]

    # The full list of chunks that were packed into the context window.
    context_chunks: list[CitedChunk]

    # Fraction of context chunks whose citation tag appears in the answer.
    # 0.0 = no chunks cited; 1.0 = every chunk cited.
    context_utilisation_ratio: float

    # Which provider and model were used for generation.
    provider: GenerationProvider
    model_name: str

    # Total tokens sent to the LLM (prompt + answer).
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# =============================================================================
# Profiling models — written to data/profiles/*.jsonl
# =============================================================================

class StageProfile(BaseModel):
    """
    One timing record written by the @profile_stage decorator.
    Each pipeline run produces many of these, one per decorated function call.
    """

    # Unique identifier for the query that triggered this pipeline run.
    # Allows grouping all stage timings for a single query.
    run_id: str

    # Name of the stage being timed (e.g. "bm25_search", "rerank", "generate").
    stage_name: str

    # ISO-8601 UTC timestamp when this stage started.
    started_at: str

    # Wall-clock duration of this stage in milliseconds.
    duration_ms: float

    # Optional: extra key-value pairs recorded alongside the timing
    # (e.g. num_candidates_in, num_candidates_out, embedding_provider).
    extra: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def now_iso() -> str:
        """Returns the current UTC time as an ISO-8601 string."""
        # datetime.utcnow() returns a naive UTC datetime; .isoformat() formats it.
        return datetime.utcnow().isoformat() + "Z"


# =============================================================================
# Evaluation models — used by eval_set.py and run_eval_matrix.py
# =============================================================================

class QueryType(str, Enum):
    """Classification of an eval query by the reasoning skill it tests."""
    FACTUAL = "factual"          # specific numbers, dates, names
    COMPARATIVE = "comparative"  # cross-company or cross-period comparison
    QUALITATIVE = "qualitative"  # summarisation, explanation, risk analysis


class EvalQuery(BaseModel):
    """
    One entry in the 50-query golden evaluation set (data/eval/eval_set.jsonl).
    Phase 7 reads this file and runs the full pipeline for each entry.
    """

    # Short alphanumeric identifier, e.g. "Q01".
    id: str

    # The natural-language question posed to the RAG pipeline.
    query: str

    # Classification: factual, comparative, or qualitative.
    query_type: QueryType

    # Optional ticker filter to pass to the retrieval stage.
    # None means "search across all companies".
    ticker: Optional[str] = None

    # Optional filing type filter (10-K, 10-Q, 8-K).
    filing_type: Optional[FilingType] = None

    # The fiscal year this query targets. None means "any year".
    fiscal_year: Optional[int] = None

    # Which document section (e.g. "Item 1A") is most likely to contain the answer.
    # Used to verify that section-aware retrieval actually surfaces the right section.
    target_section: Optional[str] = None

    # Chunk IDs (Chunk.chunk_id values) that contain the ground-truth information.
    # Populated manually after running the pipeline and inspecting results.
    relevant_chunk_ids: list[str] = Field(default_factory=list)

    # Reference answer written by a human from the actual filing text.
    # Used by the LLM judge and RAGAS context_recall metric.
    expected_answer: Optional[str] = None

    # Free-text notes for the human labeller about what to look for.
    notes: Optional[str] = None


class EvalResult(BaseModel):
    """
    Metrics produced for one EvalQuery after running the full pipeline.
    Collected by run_eval_matrix.py into the final comparison CSV.
    """

    # Which eval query this result corresponds to.
    query_id: str

    # Pipeline configuration that produced this result.
    chunking_strategy: ChunkingStrategy
    retrieval_mode: RetrievalMode
    reranker: RerankerProvider
    embedding_provider: EmbeddingProvider

    # Retrieval quality metrics.
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    precision_at_5: float
    precision_at_10: float
    mrr: float   # Mean Reciprocal Rank

    # Generation quality metrics.
    faithfulness: Optional[float] = None        # RAGAS faithfulness score
    answer_relevancy: Optional[float] = None    # RAGAS answer relevancy
    context_precision: Optional[float] = None   # RAGAS context precision
    context_recall: Optional[float] = None      # RAGAS context recall
    llm_judge_score: Optional[float] = None     # 1–5 correctness score

    # Latency breakdown (milliseconds).
    latency_embedding_ms: float = 0.0
    latency_bm25_ms: float = 0.0
    latency_vector_ms: float = 0.0
    latency_rrf_ms: float = 0.0
    latency_rerank_ms: float = 0.0
    latency_generation_ms: float = 0.0
    latency_total_ms: float = 0.0

    # Token usage in the generation stage.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_utilisation_ratio: float = 0.0
