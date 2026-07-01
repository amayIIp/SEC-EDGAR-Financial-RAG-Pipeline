# src/shared/config.py
#
# Loads configs/config.yaml into a fully-typed, validated Python object.
# Every other module imports the singleton `cfg` from here — one line,
# no YAML parsing scattered across the codebase.
#
# USAGE:
#   from src.shared.config import cfg
#   chunk_size = cfg.chunking.structure_aware.max_section_tokens
#
# OVERRIDING FOR EXPERIMENTS:
#   Pass a different config path at startup:
#     CONFIG_PATH=configs/exp_bge.yaml uvicorn src.api.main:app
#   Or from a script:
#     os.environ["CONFIG_PATH"] = "configs/exp_bge.yaml"
#     importlib.reload(src.shared.config)

from __future__ import annotations

import os            # standard-library: reads CONFIG_PATH environment variable
from functools import lru_cache  # standard-library: caches the loaded config so
                                  # YAML is only read from disk once per process
from pathlib import Path          # standard-library: filesystem path operations
from typing import Optional       # type-hint helper for nullable fields

import yaml                       # PyYAML: deserialises the YAML config file
from pydantic import BaseModel, Field  # Pydantic v2: validates the parsed dict


# =============================================================================
# Pydantic models that mirror configs/config.yaml exactly.
# If a field is present in the YAML but absent here, it is silently ignored.
# If a required field is absent from the YAML, Pydantic raises a clear error
# at startup — much better than a KeyError deep inside the pipeline.
# =============================================================================

# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestionConfig(BaseModel):
    tickers: list[str]
    form_types: list[str] = ["10-K", "10-Q"]
    years_of_history: int = 3
    requests_per_second: int = 8
    user_agent: str
    raw_data_dir: str = "data/raw"
    manifest_path: str = "data/raw/manifest.csv"
    skip_existing: bool = True


# ── Parsing ───────────────────────────────────────────────────────────────────

class ParsingConfig(BaseModel):
    parsed_data_dir: str = "data/parsed"
    bs4_parser: str = "lxml"
    section_header_regex: str
    min_content_length: int = 80
    strip_xbrl: bool = True
    boilerplate_prefixes: list[str] = Field(default_factory=list)
    parse_workers: int = 4


# ── Chunking ──────────────────────────────────────────────────────────────────

class NaiveFixedChunkingConfig(BaseModel):
    chunk_size_tokens: int = 512
    overlap_fraction: float = 0.15


class StructureAwareChunkingConfig(BaseModel):
    max_section_tokens: int = 800
    split_separators: list[str] = ["\n\n", "\n", ". "]
    table_context_sentences: int = 2
    max_table_tokens: int = 1024


class SemanticChunkingConfig(BaseModel):
    sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.50
    min_chunk_tokens: int = 100
    max_chunk_tokens: int = 900
    sentence_encode_batch_size: int = 64


class ChunkingConfig(BaseModel):
    chunks_dir: str = "data/chunks"
    token_encoding: str = "cl100k_base"
    naive_fixed: NaiveFixedChunkingConfig = Field(default_factory=NaiveFixedChunkingConfig)
    structure_aware: StructureAwareChunkingConfig = Field(
        default_factory=StructureAwareChunkingConfig
    )
    semantic: SemanticChunkingConfig = Field(default_factory=SemanticChunkingConfig)


# ── Embedding ─────────────────────────────────────────────────────────────────

class OpenAIEmbeddingConfig(BaseModel):
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 512
    retry_base_delay: float = 1.0
    max_retries: int = 5


class BGEEmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-large-en-v1.5"
    dimensions: int = 1024
    batch_size: int = 32
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    device: str = "auto"


class EmbeddingConfig(BaseModel):
    provider: str = "openai"           # "openai" | "bge"
    openai: OpenAIEmbeddingConfig = Field(default_factory=OpenAIEmbeddingConfig)
    bge: BGEEmbeddingConfig = Field(default_factory=BGEEmbeddingConfig)


# ── OpenSearch ────────────────────────────────────────────────────────────────

class OpenSearchConfig(BaseModel):
    host: str = "localhost"
    port: int = 9200
    index_name: str = "sec_chunks_v1"
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    shards: int = 1
    replicas: int = 0
    bulk_batch_size: int = 500
    bulk_timeout: str = "60s"


# ── Qdrant ────────────────────────────────────────────────────────────────────

class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "sec_chunks_openai_v1"
    distance: str = "Cosine"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100
    hnsw_ef: int = 128
    upsert_batch_size: int = 256


# ── Retrieval ─────────────────────────────────────────────────────────────────

class DefaultFiltersConfig(BaseModel):
    ticker: Optional[str] = None
    filing_type: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class RetrievalConfig(BaseModel):
    mode: str = "hybrid"
    retrieval_depth: int = 50
    rrf_k: int = 60
    pre_rerank_top_k: int = 50
    default_filters: DefaultFiltersConfig = Field(default_factory=DefaultFiltersConfig)


# ── Reranking ─────────────────────────────────────────────────────────────────

class CohereRerankerConfig(BaseModel):
    model: str = "rerank-english-v3.0"
    max_docs_per_call: int = 200


class BGERerankerConfig(BaseModel):
    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 16
    device: str = "auto"


class RerankingConfig(BaseModel):
    provider: str = "cohere"           # "cohere" | "bge" | "none"
    top_n: int = 8
    cohere: CohereRerankerConfig = Field(default_factory=CohereRerankerConfig)
    bge: BGERerankerConfig = Field(default_factory=BGERerankerConfig)


# ── Generation ────────────────────────────────────────────────────────────────

class OpenAIGenerationConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 30


class AnthropicGenerationConfig(BaseModel):
    model: str = "claude-3-haiku-20240307"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 30


class OllamaGenerationConfig(BaseModel):
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    num_predict: int = 1024
    timeout: int = 120


class GenerationConfig(BaseModel):
    provider: str = "openai"
    use_context_compression: bool = False
    dedup_similarity_threshold: float = 0.92
    openai: OpenAIGenerationConfig = Field(default_factory=OpenAIGenerationConfig)
    anthropic: AnthropicGenerationConfig = Field(default_factory=AnthropicGenerationConfig)
    ollama: OllamaGenerationConfig = Field(default_factory=OllamaGenerationConfig)
    compression_model: str = "gpt-4o-mini"
    compression_target_ratio: float = 0.40


# ── Evaluation ────────────────────────────────────────────────────────────────

class JudgeConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    min_passing_score: int = 3


class EvaluationConfig(BaseModel):
    eval_set_path: str = "data/eval/eval_set.jsonl"
    results_dir: str = "data/eval/results"
    recall_k_values: list[int] = [5, 10, 20]
    min_recall_at_10: float = 0.60
    min_faithfulness: float = 0.70
    min_answer_relevancy: float = 0.65
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    run_ragas: bool = True
    run_llm_judge: bool = True


# ── Profiling ─────────────────────────────────────────────────────────────────

class ProfilingConfig(BaseModel):
    profiles_dir: str = "data/profiles"
    enabled: bool = True
    exact_match_cache: bool = True
    semantic_cache_threshold: float = 0.97
    semantic_cache_max_entries: int = 1000
    pool_size_candidates: list[int] = [50, 25, 15, 10]


# ── API ───────────────────────────────────────────────────────────────────────

class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    cors_origins: list[str] = ["*"]
    request_timeout: int = 60


# =============================================================================
# Root config object — contains one instance of each sub-config above
# =============================================================================

class PipelineConfig(BaseModel):
    """
    Root configuration object for the entire pipeline.
    Loaded once at process startup; accessed everywhere via the `cfg` singleton.
    """
    ingestion: IngestionConfig
    parsing: ParsingConfig
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    opensearch: OpenSearchConfig = Field(default_factory=OpenSearchConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    api: APIConfig = Field(default_factory=APIConfig)


# =============================================================================
# Loader — reads the YAML file and returns a validated PipelineConfig
# =============================================================================

@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> PipelineConfig:
    """
    Reads the YAML config file from disk and returns a validated PipelineConfig.

    The result is cached via @lru_cache so the file is only read once per
    process lifetime.  Call `load_config.cache_clear()` in tests that need to
    swap configs between test functions.

    Priority order for config path:
      1. `config_path` argument (explicit override in tests or scripts)
      2. CONFIG_PATH environment variable
      3. Default: "configs/config.yaml"
    """
    # Determine the path to use from the priority chain described above.
    path_str = config_path or os.environ.get("CONFIG_PATH", "configs/config.yaml")
    path = Path(path_str)

    # Fail loudly at startup if the config file is missing — better than a
    # confusing KeyError or AttributeError deep inside the pipeline later.
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline config not found at '{path.resolve()}'. "
            "Copy configs/config.yaml to the expected path and fill in your values."
        )

    # Read the entire YAML file into memory and parse it into a Python dict.
    raw_yaml = path.read_text(encoding="utf-8")
    raw_dict = yaml.safe_load(raw_yaml)

    # Validate and coerce the raw dict into our typed PipelineConfig object.
    # Pydantic raises ValidationError with field-level messages if anything is wrong.
    return PipelineConfig(**raw_dict)


# Module-level singleton — import this everywhere else:
#   from src.shared.config import cfg
# This runs exactly once when the module is first imported.
cfg: PipelineConfig = load_config()
