from __future__ import annotations 
import json 
from unittest.mock import MagicMock, patch 
import pytest 
from src.evaluation.llm_judge import LLMJudge 
from src.evaluation.retrieval_metrics import calculate_mrr, calculate_precision_at_k, calculate_recall_at_k 
def test_calculate_recall_at_k() -> None:
    """
    Verifies that calculate_recall_at_k correctly computes the proportion of ground-truth
    chunks retrieved in the top-k results.
    """
    relevant = ["doc1", "doc2", "doc3"]
    retrieved_all = ["doc1", "other1", "doc2", "other2", "doc3"]
    recall_all = calculate_recall_at_k(retrieved_all, relevant, k=5)
    assert recall_all == 1.0, "Expected 100% recall."
    retrieved_partial = ["doc1", "other1", "doc2", "other2", "doc3"]
    recall_partial = calculate_recall_at_k(retrieved_partial, relevant, k=3)
    assert recall_partial == 2.0 / 3.0, "Recall calculation is incorrect."
    assert calculate_recall_at_k(retrieved_all, [], k=5) == 1.0
def test_calculate_precision_at_k() -> None:
    """
    Verifies that calculate_precision_at_k correctly computes the proportion of retrieved
    chunks that are actually relevant in the top-k slice.
    """
    relevant = ["doc1", "doc2"]
    retrieved = ["doc1", "other1", "doc2", "other2"]
    precision = calculate_precision_at_k(retrieved, relevant, k=3)
    assert precision == 2.0 / 3.0, "Precision calculation is incorrect."
    precision_1 = calculate_precision_at_k(retrieved, relevant, k=1)
    assert precision_1 == 1.0
    assert calculate_precision_at_k(retrieved, relevant, k=0) == 0.0
def test_calculate_mrr() -> None:
    """
    Verifies that calculate_mrr calculates the reciprocal rank of the first relevant document correctly.
    """
    relevant = ["doc1", "doc2"]
    retrieved_1 = ["doc1", "other1", "doc2"]
    assert calculate_mrr(retrieved_1, relevant) == 1.0
    retrieved_3 = ["other1", "other2", "doc2", "doc1"]
    assert calculate_mrr(retrieved_3, relevant) == 1.0 / 3.0
    retrieved_none = ["other1", "other2"]
    assert calculate_mrr(retrieved_none, relevant) == 0.0
@patch("src.evaluation.llm_judge.OpenAI") 
def test_llm_judge_grader(mock_openai_class: MagicMock) -> None:
    """
    Verifies that LLMJudge formats prompts, queries OpenAI, and parses the structured response.
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = json.dumps({
        "score": 4,
        "justification": "The answer is mostly complete and accurate."
    })
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    judge = LLMJudge()
    result = judge.grade_answer(
        query="What is the net profit?",
        answer="The net profit is $5B.",
        expected_answer="Net profit is $5.2B."
    )
    assert result["score"] == 4, "LLM Judge returned incorrect score."
    assert "mostly complete" in result["justification"], "LLM Judge returned incorrect justification."
    mock_client.chat.completions.create.assert_called_once()
