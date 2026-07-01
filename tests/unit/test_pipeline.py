# tests/unit/test_pipeline.py
# This module implements unit tests for the pipeline orchestrators (SECRAGPipeline, PipelineRunner)
# and the LLM generation client (SECGenerator). We mock network I/O, database indices,
# and LLM clients to verify end-to-end data flow orchestration and output formatting.

from __future__ import annotations # Allow self-referencing type annotations.
from unittest.mock import AsyncMock, MagicMock, patch # Mocking utilities.
import pytest # Testing framework.
from src.generation.llm_client import SECGenerator # Subject under test.
from src.pipeline import SECRAGPipeline # Subject under test.
from src.api.pipeline_runner import PipelineRunner # Subject under test.
from src.shared.models import CitedChunk, FilingType, GenerationOutput, RankedResult, Chunk, ChunkingStrategy, BM25Result, VectorResult # Models.


# Helper to build a mock RankedResult record for testing.
def _make_mock_ranked_result(chunk_id: str) -> RankedResult:
    """Helper to build a dummy RankedResult record for testing."""
    chunk = Chunk(
        chunk_id=chunk_id,
        text="Sample context for LLM question answering.",
        strategy=ChunkingStrategy.STRUCTURE_AWARE,
        chunk_index=0,
        is_table=False,
        ticker="AAPL",
        cik="0000320193",
        filing_type=FilingType.TEN_K,
        filing_date="2023-09-30",
        fiscal_year=2023,
        accession_number="0000320193-23-000106",
        section="Item 7",
        section_title="MD&A",
        token_count=10
    )
    return RankedResult(
        chunk=chunk,
        rerank_score=0.95,
        final_rank=1,
        rrf_score=0.08
    )


@patch("src.generation.llm_client.OpenAI") # Mock the OpenAI API client class.
def test_sec_generator_openai_routing(mock_openai_class: MagicMock) -> None:
    """
    Verifies that SECGenerator properly formats context prompts, routes requests to OpenAI,
    and returns parsed GenerationOutput records.
    """
    # Setup mock client.
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Configure mock completion return data.
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "According to the financial statements, R&D expense was $29.9B [1]."
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    # Set mock usage metadata.
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 80
    mock_usage.completion_tokens = 25
    mock_response.usage = mock_usage
    
    mock_client.chat.completions.create.return_value = mock_response

    # Instantiate generator.
    generator = SECGenerator()
    
    # Setup mock cited chunks.
    ranked_result = _make_mock_ranked_result("chunk_1")
    cited_chunk = CitedChunk(
        ranked_result=ranked_result,
        citation_tag="[1]",
        packed_text="Filing Source 1 | Apple Inc. | R&D Expense: $29.9B",
        packed_token_count=20
    )
    
    # Execute generation.
    output = generator.generate(
        query="What was the R&D expense?",
        chunks=[cited_chunk],
        provider="openai"
    )
    
    # Assertions.
    assert isinstance(output, GenerationOutput)
    assert output.answer == "According to the financial statements, R&D expense was $29.9B [1]."
    assert output.prompt_tokens == 80
    assert output.completion_tokens == 25
    assert output.cited_tags == ["[1]"]


@patch("src.pipeline.FilingManifest") # Mock the FilingManifest database.
@patch("src.pipeline.OpenSearchIndexManager") # Mock OpenSearch manager.
@patch("src.pipeline.QdrantIndexManager") # Mock Qdrant manager.
@patch("src.pipeline.SECEmbedder") # Mock embedder.
@patch("src.pipeline.fetch_filing_index") # Mock index fetcher function.
@patch("src.pipeline.download_filing_html") # Mock download html function.
@patch("src.pipeline.parse_filing") # Mock parse filing function.
def test_pipeline_ingest_orchestration(
    mock_parse_filing: MagicMock,
    mock_download_html: MagicMock,
    mock_fetch_index: MagicMock,
    mock_embedder_class: MagicMock,
    mock_qdrant_manager_class: MagicMock,
    mock_opensearch_manager_class: MagicMock,
    mock_manifest_class: MagicMock
) -> None:
    """
    Verifies that SECRAGPipeline coordinates all sub-components (download, parse,
    chunk, embed, and index) correctly during ingestion.
    """
    # Configure mock return values.
    # Set mock manifest to say the accession does not exist yet.
    mock_manifest = MagicMock()
    mock_manifest.exists.return_value = False
    mock_manifest_class.return_value = mock_manifest
    
    # Set mock filing list.
    mock_fetch_index.return_value = [
        {
            "accession_number": "acc_123",
            "form": "10-K",
            "filing_date": "2023-09-30",
            "fiscal_year": 2023,
            "primary_doc": "aapl.htm"
        }
    ]
    
    # Mock download success.
    mock_download_html.return_value = True
    
    # Mock parsing success with a real ParsedFiling model to prevent Pydantic validation failures.
    from src.shared.models import ParsedFiling, ParsedSection, FilingMetadata, FilingType
    metadata = FilingMetadata(
        ticker="AAPL",
        cik="0000320193",
        filing_type=FilingType.TEN_K,
        filing_date="2023-09-30",
        fiscal_year=2023,
        accession_number="acc_123",
        local_path="data/raw/AAPL/10-K_2023-09-30.html"
    )
    sections = [
        ParsedSection(
            item_id="Item 1",
            title="Business",
            content="Operating margin expanded to 30 percent due to growth in services revenue.",
            tables=[]
        )
    ]
    mock_parsed_filing = ParsedFiling(metadata=metadata, sections=sections)
    mock_parse_filing.return_value = mock_parsed_filing
    
    # Mock embedder.
    mock_embedder = MagicMock()
    mock_embedder.get_dimension.return_value = 1536
    mock_embedder.embed.return_value = [[0.1] * 1536]
    mock_embedder_class.return_value = mock_embedder

    # Instantiate pipeline.
    pipeline = SECRAGPipeline()
    
    # Run pipeline ingestion.
    indexed_count = pipeline.ingest_ticker_data(
        ticker="AAPL",
        forms=["10-K"],
        limit=1,
        embedding_provider="openai"
    )
    
    # Verify that the pipeline indexed the chunk.
    assert indexed_count > 0, "No chunks were indexed."
    mock_download_html.assert_called_once()
    mock_parse_filing.assert_called_once()


@pytest.mark.anyio # Mark as async test.
@patch("src.api.pipeline_runner.BGEReranker") # Mock the local CrossEncoder class.
@patch("src.api.pipeline_runner.CohereReranker") # Mock the Cohere class.
@patch("src.api.pipeline_runner.SECGenerator") # Mock the generator class.
@patch("src.api.pipeline_runner.bm25_search", new_callable=AsyncMock) # Mock async BM25 search.
@patch("src.api.pipeline_runner.vector_search", new_callable=AsyncMock) # Mock async vector search.
@patch("src.api.pipeline_runner.SECEmbedder") # Mock embedder class.
async def test_pipeline_runner_run_pipeline(
    mock_embedder_class: MagicMock,
    mock_vector_search: AsyncMock,
    mock_bm25_search: AsyncMock,
    mock_generator_class: MagicMock,
    mock_cohere_class: MagicMock,
    mock_bge_class: MagicMock
) -> None:
    """
    Verifies that PipelineRunner runs the complete run_pipeline query sequence
    (retrieval, fusion, reranking, packing, and generation) and returns correct output.
    """
    # Setup mock embedder.
    mock_embedder = MagicMock()
    mock_embedder_class.return_value = mock_embedder
    mock_embedder.embed.return_value = [[0.1] * 1536]
    
    # Mock class initializations to prevent loading real models or setting API connections.
    mock_generator = MagicMock()
    mock_generator_class.return_value = mock_generator
    
    mock_cohere = MagicMock()
    mock_cohere_class.return_value = mock_cohere
    
    mock_bge = MagicMock()
    mock_bge_class.return_value = mock_bge

    # Setup mock search responses.
    chunk_a = _make_mock_ranked_result("chunk_a").chunk
    mock_bm25_search.return_value = [BM25Result(chunk=chunk_a, bm25_score=10.0, bm25_rank=1)]
    mock_vector_search.return_value = [VectorResult(chunk=chunk_a, vector_score=0.9, vector_rank=1)]

    # Instantiate the runner under test.
    runner = PipelineRunner()
    
    # Mock reranker to return mock ranked results.
    mock_cohere.rerank.return_value = [_make_mock_ranked_result("chunk_a")]
    
    # Mock generator to return mock generation output.
    mock_gen_output = GenerationOutput(
        query="What is profit?",
        answer="The profit is $5B.",
        cited_tags=["[1]"],
        context_chunks=[
            CitedChunk(
                ranked_result=_make_mock_ranked_result("chunk_a"),
                citation_tag="[1]",
                packed_text="Packed text",
                packed_token_count=10
            )
        ],
        context_utilisation_ratio=1.0,
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
        provider="openai",
        model_name="gpt-4o-mini"
    )
    mock_generator.generate.return_value = mock_gen_output
    
    # Execute runner.
    result = await runner.run_pipeline(
        query="What is profit?",
        mode="hybrid",
        top_k=5,
        top_n=2,
        reranker_provider="cohere"
    )
    
    # Assertions.
    assert result["answer"] == "The profit is $5B."
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == "chunk_a"
    assert result["total_tokens"] == 70 # prompt_tokens + completion_tokens.
    assert "raw_state" in result


@patch("src.pipeline.SECRAGPipeline") # Mock the SECRAGPipeline class.
def test_pipeline_runner_ingest(mock_pipeline_class: MagicMock) -> None:
    """
    Verifies that PipelineRunner.ingest instantiates the pipeline and calls ingest_ticker_data.
    """
    # Create mock pipeline instance.
    mock_pipeline = MagicMock()
    mock_pipeline_class.return_value = mock_pipeline
    mock_pipeline.ingest_ticker_data.return_value = 42 # Mock indexed chunk count.
    
    # Instantiate the runner.
    runner = PipelineRunner()
    
    # Run the ingestion method.
    count = runner.ingest(
        ticker="AAPL",
        forms=["10-K"],
        limit=3,
        embedding_provider="openai"
    )
    
    # Assertions.
    assert count == 42, "Returned chunk count mismatch."
    mock_pipeline.ingest_ticker_data.assert_called_once_with(
        ticker="AAPL",
        forms=["10-K"],
        limit=3,
        embedding_provider="openai"
    )
