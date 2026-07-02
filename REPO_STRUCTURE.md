# Repository Structure — SEC EDGAR Financial RAG Pipeline

This document is the **canonical directory reference** for the project.
Every directory and file listed here has a specific, non-overlapping
responsibility. Read this before adding a new file anywhere.

---

## Top-Level Layout

```
sec-rag-pipeline/
├── configs/                  # All YAML config files — the single source of truth for tunable params
│   └── config.yaml           # Master config (chunk size, model names, thresholds, …)
│
├── data/                     # All pipeline data artefacts — NEVER commit large files; use .gitkeep
│   ├── raw/                  # Raw HTML filings from SEC EDGAR, one file per submission
│   │   └── {ticker}/         # e.g. data/raw/AAPL/10-K_2023-09-30.html
│   ├── parsed/               # Structured JSON output from the HTML parser
│   │   └── {ticker}/         # e.g. data/parsed/AAPL/10-K_2023-09-30.json
│   ├── chunks/               # Chunked output, one sub-folder per strategy
│   │   ├── naive_fixed/      # data/chunks/naive_fixed/chunks.jsonl
│   │   ├── structure_aware/  # data/chunks/structure_aware/chunks.jsonl
│   │   └── semantic/         # data/chunks/semantic/chunks.jsonl
│   ├── indexes/              # Persisted index artefacts (Qdrant snapshots, OS index exports)
│   ├── eval/                 # Evaluation artefacts
│   │   ├── eval_set.jsonl    # 50-query golden eval set (query + expected chunks + expected answer)
│   │   └── results/          # Per-run eval output CSVs and markdown tables
│   ├── profiles/             # Profiling JSON-lines logs (one file per pipeline run)
│   └── results/              # Final consolidated benchmark tables (Phase 8 output)
│
├── docker/                   # Docker-specific files kept separate from project root
│   ├── docker-compose.yml    # Spins up OpenSearch + Qdrant (+ optional FastAPI/Streamlit)
│   ├── opensearch.env        # OpenSearch environment overrides
│   └── qdrant.env            # Qdrant environment overrides
│
├── docs/                     # Design docs and ADRs — not user docs (those go in README)
│   ├── architecture.md       # Pipeline diagram description + data-flow narrative
│   └── decisions.md          # Architecture Decision Records (ADRs): why each tech was chosen
│
├── notebooks/                # Exploratory Jupyter notebooks — one per investigation topic
│   ├── 00_eda.ipynb          # Corpus EDA: filing counts, section distributions, table density
│   ├── 01_chunking_comparison.ipynb   # Visual comparison of chunker outputs
│   ├── 02_retrieval_analysis.ipynb    # Recall curves, RRF sensitivity, reranker deltas
│   └── 03_eval_results.ipynb          # Final metric tables and visualisations for write-up
│
├── scripts/                  # Thin shell wrappers that call Python CLI entry-points
│   ├── download_tickers.sh   # Calls src/ingestion/run_ingest.py for a ticker list
│   ├── run_pipeline_local.sh # End-to-end local run: ingest → parse → chunk → index → eval
│   └── run_eval.sh           # Full eval matrix run (Phase 7)
│
├── src/                      # All Python source code — pure library modules, no __main__ logic
│   │                         # except in run_*.py entry-points
│   │
│   ├── shared/               # Cross-cutting concerns used by every other module
│   │   ├── __init__.py
│   │   ├── config.py         # Load configs/config.yaml → typed Pydantic Settings object
│   │   ├── logging_setup.py  # Structured JSON logging; propagates request-id through log records
│   │   └── models.py         # Shared dataclasses: Chunk, SearchResult, RankedResult, GenerationOutput
│   │
│   ├── ingestion/            # Phase 1a — Downloading raw filings from SEC EDGAR
│   │   ├── __init__.py
│   │   ├── edgar_client.py   # Rate-limited HTTP client (10 req/s, required User-Agent header)
│   │   ├── manifest.py       # Manifest CSV: track company/ticker/form/date/accession/path
│   │   └── run_ingest.py     # CLI entry-point: parse args, iterate tickers, call edgar_client
│   │
│   ├── parsing/              # Phase 1b — Converting raw HTML → structured section JSON
│   │   ├── __init__.py
│   │   ├── html_parser.py        # BeautifulSoup/lxml orchestrator; produces the section schema
│   │   ├── table_extractor.py    # Isolates <table> tags → {caption, rows[][], raw_text}
│   │   ├── boilerplate_filter.py # Strips TOC, XBRL inline tags, page header/footer repetitions
│   │   ├── run_parse.py          # CLI entry-point: batch-parse all raw/ files → parsed/ JSON
│   │   └── validate_parse.py     # Validation report: failure rate, avg sections, random samples
│   │
│   ├── chunking/             # Phase 2 — Splitting parsed JSON into indexable Chunk objects
│   │   ├── __init__.py
│   │   ├── base_chunker.py           # Abstract BaseChunker + Chunk dataclass definition
│   │   ├── naive_fixed_chunker.py    # 512-token fixed chunks, 15% overlap (tiktoken)
│   │   ├── structure_aware_chunker.py # Section-boundary first, recursive paragraph fallback
│   │   ├── semantic_chunker.py       # Cosine-similarity boundary detection (all-MiniLM-L6-v2)
│   │   └── run_chunk.py              # CLI entry-point: run all 3 chunkers, write JSONL, print stats
│   │
│   ├── indexing/             # Phase 3 — Embedding + loading chunks into search backends
│   │   ├── __init__.py
│   │   ├── embedder.py           # Embedding interface: OpenAI text-embedding-3-small + BGE-large
│   │   ├── opensearch_index.py   # Create OS index mapping, bulk-index chunks, idempotent upsert
│   │   ├── qdrant_index.py       # Create Qdrant collection, upsert point batches, idempotent
│   │   ├── run_index.py          # CLI entry-point: embed + index a named chunking strategy
│   │   └── smoke_test.py         # One BM25 query + one vector query; print top-5 to confirm
│   │
│   ├── retrieval/            # Phase 4 — Querying both indexes and fusing results
│   │   ├── __init__.py
│   │   ├── bm25_search.py    # Async BM25 search: OpenSearch query + metadata filter construction
│   │   ├── vector_search.py  # Async vector search: Qdrant query + payload filter construction
│   │   ├── rrf_fusion.py     # ~15-line Reciprocal Rank Fusion over two ranked lists
│   │   ├── hybrid_search.py  # Unified interface: bm25_only | vector_only | hybrid + asyncio.gather
│   │   └── run_search_cli.py # CLI: query + mode → top-10 results in a readable table
│   │
│   ├── reranking/            # Phase 5 — Cross-encoder reranking of fused candidates
│   │   ├── __init__.py
│   │   ├── base_reranker.py          # Abstract BaseReranker interface
│   │   ├── cohere_reranker.py        # Cohere rerank-english-v3.0 (handles doc-count batching)
│   │   ├── bge_reranker.py           # BAAI/bge-reranker-v2-m3 CrossEncoder (GPU if available)
│   │   └── run_rerank_compare.py     # Side-by-side comparison per query + Kendall-tau metric
│   │
│   ├── generation/           # Phase 6 — Context packing + LLM answer generation
│   │   ├── __init__.py
│   │   ├── context_packer.py     # Deduplicate chunks, order by score, label [1]…[n]
│   │   ├── prompt_templates.py   # All prompt strings: cite-or-decline, compression, LLM-judge rubric
│   │   ├── llm_client.py         # Generation backends: OpenAI/Claude API + Ollama Llama 3.1 8B
│   │   ├── context_compressor.py # Optional: summarise each chunk to ~40% via cheap LLM call
│   │   └── citation_checker.py   # Parse [n] bracket tags → context utilisation ratio
│   │
│   ├── evaluation/           # Phase 7 — Systematic quality measurement
│   │   ├── __init__.py
│   │   ├── eval_set.py           # 50-query JSONL golden set + generation helper
│   │   ├── retrieval_metrics.py  # Recall@k, Precision@k, MRR against labelled chunk IDs
│   │   ├── ragas_wrapper.py      # Adapt pipeline output to RAGAS Dataset format, run metrics
│   │   ├── llm_judge.py          # LLM-as-judge: 1–5 correctness score with structured JSON output
│   │   └── run_eval_matrix.py    # Master script: all combinations → CSV + markdown comparison table
│   │
│   ├── profiling/            # Phase 8 — Latency measurement and optimisation experiments
│   │   ├── __init__.py
│   │   ├── stage_profiler.py         # @profile_stage decorator + ProfilerContext → JSON-lines log
│   │   ├── latency_report.py         # p50/p95/p99 per stage from profile logs; bottleneck ID
│   │   ├── cache.py                  # Exact-match + semantic query cache (dict + cosine threshold)
│   │   ├── context_window_analysis.py # Token usage + cited-vs-packed utilisation ratio
│   │   └── run_optimization_bench.py  # A/B: baseline vs each optimisation flag → consolidated table
│   │
│   ├── api/                  # Phase 9a — FastAPI service layer
│   │   ├── __init__.py
│   │   ├── main.py           # App factory, mounts routers, configures middleware
│   │   ├── schemas.py        # Pydantic v2 request/response models for all endpoints
│   │   ├── pipeline_runner.py # Async orchestrator: wires retrieval → rerank → generation
│   │   └── error_handlers.py # Global exception → structured JSON error response
│   │
│   └── demo/                 # Phase 9b — Streamlit frontend
│       ├── __init__.py
│       └── streamlit_app.py  # Query UI, citations panel, latency chart, mode A/B comparison
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_chunkers.py          # Parametrised tests for all three chunking strategies
│   │   ├── test_rrf.py               # Edge cases: empty lists, ties, k sensitivity
│   │   └── test_citation_checker.py  # Bracket parsing with malformed/missing tags
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_indexing_round_trip.py  # Index 10 chunks → query → assert top-1 hit match
│   │   └── test_hybrid_search.py        # BM25 + vector + RRF end-to-end with real indexes
│   └── e2e/
│       ├── __init__.py
│       └── test_api_endpoints.py     # Spin up TestClient, POST /query, assert response schema
│
├── .env.example              # Template for required env vars; never commit a real .env
├── .gitignore                # data/raw/, data/parsed/, *.jsonl, __pycache__, .env, *.pyc
├── Dockerfile                # Production image for FastAPI service
├── REPO_STRUCTURE.md         # THIS FILE — canonical directory reference
├── README.md                 # User-facing project summary, results, setup instructions
├── configs/config.yaml       # Master YAML config (all tunable parameters — see Phase 0)
├── pyproject.toml            # Build metadata, tool config (black, ruff, mypy, pytest)
└── requirements.txt          # Pinned dependencies grouped by pipeline phase
```

---

## Key Design Principles

| Principle | How It's Applied |
|-----------|-----------------|
| **One responsibility per file** | `html_parser.py` orchestrates; `table_extractor.py` and `boilerplate_filter.py` are called by it — never the reverse |
| **Entry-points are thin** | `run_*.py` files do arg-parsing and logging only; all logic lives in the library module |
| **Config is the single source of truth** | Every tunable value is in `config.yaml`; modules read it via `src/shared/config.py` — no magic constants scattered in source files |
| **Shared models prevent drift** | `SearchResult`, `Chunk`, `RankedResult` are defined once in `src/shared/models.py`; all modules import from there |
| **Data never in source control** | `data/` is in `.gitignore`; only `.gitkeep` files are committed to preserve the directory skeleton |
| **Tests mirror source structure** | `tests/unit/test_chunkers.py` tests `src/chunking/*`; integration tests require live Docker services |

---

## Data Flow Summary

```
SEC EDGAR API
    │  (rate-limited HTTP, src/ingestion/)
    ▼
data/raw/{ticker}/*.html
    │  (BeautifulSoup + lxml, src/parsing/)
    ▼
data/parsed/{ticker}/*.json          ← {sections[], tables[]}
    │  (3 chunker strategies, src/chunking/)
    ▼
data/chunks/{strategy}/chunks.jsonl  ← {chunk_id, text, metadata}
    │  (embed + bulk index, src/indexing/)
    ├──────────────────────────────────┐
    ▼                                  ▼
OpenSearch index (BM25)          Qdrant collection (vectors)
    │                                  │
    └──────────┬───────────────────────┘
               │  (asyncio.gather, src/retrieval/)
               ▼
         RRF-fused candidates          ← top-50 from each, fused to top-50
               │  (CrossEncoder, src/reranking/)
               ▼
         Reranked top-8 chunks
               │  (context packing + LLM call, src/generation/)
               ▼
         {answer, citations[], latency_breakdown}
               │  (FastAPI, src/api/)
               ▼
         POST /query  →  Streamlit (src/demo/)
```
