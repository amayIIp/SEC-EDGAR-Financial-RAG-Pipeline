# tests/unit/test_evaluation.py
# This module implements unit tests for the retrieval metrics and LLM judge components.
# We verify the arithmetic calculations for Recall@k, Precision@k, and Mean Reciprocal Rank (MRR),
# and mock the OpenAI client to test LLM judge grading.

from __future__ import annotations # Allow self-referencing type annotations.
import json # Standard library module to parse JSON.
from unittest.mock import MagicMock, patch # Mocking tools.
import pytest # Testing framework.
from src.evaluation.llm_judge import LLMJudge # Subject under test.
from src.evaluation.retrieval_metrics import calculate_mrr, calculate_precision_at_k, calculate_recall_at_k # Subjects under test.


def test_calculate_recall_at_k() -> None:
    """
    Verifies that calculate_recall_at_k correctly computes the proportion of ground-truth
    chunks retrieved in the top-k results.
    """
    # Define ground-truth relevant chunk IDs.
    relevant = ["doc1", "doc2", "doc3"]
    
    # Case 1: All relevant documents retrieved in top 5.
    retrieved_all = ["doc1", "other1", "doc2", "other2", "doc3"]
    recall_all = calculate_recall_at_k(retrieved_all, relevant, k=5)
    assert recall_all == 1.0, "Expected 100% recall."
    
    # Case 2: Only 2 out of 3 relevant documents retrieved in top 3.
    retrieved_partial = ["doc1", "other1", "doc2", "other2", "doc3"]
    # At k=3, the retrieved slice is ["doc1", "other1", "doc2"].
    # Intersects with ground-truth at {"doc1", "doc2"} (2 matches).
    recall_partial = calculate_recall_at_k(retrieved_partial, relevant, k=3)
    assert recall_partial == 2.0 / 3.0, "Recall calculation is incorrect."
    
    # Case 3: Empty relevant list.
    assert calculate_recall_at_k(retrieved_all, [], k=5) == 1.0


def test_calculate_precision_at_k() -> None:
    """
    Verifies that calculate_precision_at_k correctly computes the proportion of retrieved
    chunks that are actually relevant in the top-k slice.
    """
    relevant = ["doc1", "doc2"]
    
    # Case 1: At k=3, retrieved has 2 relevant out of 3.
    retrieved = ["doc1", "other1", "doc2", "other2"]
    # The top-3 slice is ["doc1", "other1", "doc2"] (2 matches out of 3 total elements).
    precision = calculate_precision_at_k(retrieved, relevant, k=3)
    assert precision == 2.0 / 3.0, "Precision calculation is incorrect."
    
    # Case 2: At k=1, top hit is relevant.
    precision_1 = calculate_precision_at_k(retrieved, relevant, k=1)
    assert precision_1 == 1.0
    
    # Case 3: Invalid k value.
    assert calculate_precision_at_k(retrieved, relevant, k=0) == 0.0


def test_calculate_mrr() -> None:
    """
    Verifies that calculate_mrr calculates the reciprocal rank of the first relevant document correctly.
    """
    relevant = ["doc1", "doc2"]
    
    # Case 1: First relevant document appears at rank 1.
    retrieved_1 = ["doc1", "other1", "doc2"]
    assert calculate_mrr(retrieved_1, relevant) == 1.0
    
    # Case 2: First relevant document appears at rank 3.
    retrieved_3 = ["other1", "other2", "doc2", "doc1"]
    # Rank of doc2 is 3; reciprocal rank = 1/3.
    assert calculate_mrr(retrieved_3, relevant) == 1.0 / 3.0
    
    # Case 3: No relevant documents found.
    retrieved_none = ["other1", "other2"]
    assert calculate_mrr(retrieved_none, relevant) == 0.0


@patch("src.evaluation.llm_judge.OpenAI") # Mock the OpenAI class.
def test_llm_judge_grader(mock_openai_class: MagicMock) -> None:
    """
    Verifies that LLMJudge formats prompts, queries OpenAI, and parses the structured response.
    """
    # Create mock client.
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Mock chat completion return data.
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    # Return structured JSON inside the content of the message.
    mock_message.content = json.dumps({
        "score": 4,
        "justification": "The answer is mostly complete and accurate."
    })
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    # Instantiate judge.
    judge = LLMJudge()
    
    # Execute grading.
    result = judge.grade_answer(
        query="What is the net profit?",
        answer="The net profit is $5B.",
        expected_answer="Net profit is $5.2B."
    )
    
    # Verify score and justification.
    assert result["score"] == 4, "LLM Judge returned incorrect score."
    assert "mostly complete" in result["justification"], "LLM Judge returned incorrect justification."
    # Verify OpenAI was called.
    mock_client.chat.completions.create.assert_called_once()
