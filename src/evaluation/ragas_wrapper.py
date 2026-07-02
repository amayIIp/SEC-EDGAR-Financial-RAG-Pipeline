# src/evaluation/ragas_wrapper.py
# This module implements the RAGAS evaluation runner.
# RAGAS (Retrieval Augmented Generation Assessment) uses LLM-as-a-judge to evaluate
# answer faithfulness, relevancy, and context metrics.

from __future__ import annotations # Allow self-referencing type annotations.
import os # Standard library module to check environment keys.
from typing import Any, Dict, List, Optional # Type helpers.
from src.shared.config import cfg # Config settings loader.
from src.shared.logging_setup import get_logger # Logger.

log = get_logger(__name__)

# Conditionally import RAGAS to prevent imports crashes when running without the libraries.
try:
    from ragas import evaluate as ragas_evaluate # Core evaluator.
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall # Core metrics.
    from datasets import Dataset # Hugging Face Dataset builder.
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def run_ragas_evaluation(
    eval_inputs: List[Dict[str, Any]]
) -> Optional[Dict[str, float]]:
    """
    Runs automated RAGAS metrics over a list of evaluated pipeline outputs.
    Each input dict must contain: question, answer, contexts (list of strings), and ground_truth.
    """
    # Verify package availability.
    if not RAGAS_AVAILABLE:
        log.warning("ragas_library_unavailable", action="skipping_ragas_eval")
        return None

    # RAGAS requires an active OpenAI API key for its LLM grading prompts.
    if not os.getenv("OPENAI_API_KEY"):
        log.warning("openai_key_missing_for_ragas", action="skipping_ragas_eval")
        return None

    log.info("starting_ragas_evaluation", count=len(eval_inputs))

    # Step 1: Format data arrays for Hugging Face Dataset.
    # RAGAS expects lists of strings for question, answer, and ground_truths,
    # and a list of lists of strings for contexts.
    questions = [item["question"] for item in eval_inputs]
    answers = [item["answer"] for item in eval_inputs]
    contexts = [item["contexts"] for item in eval_inputs]
    # RAGAS expects ground_truths to be a list of lists of strings.
    ground_truths = [[item["ground_truth"]] for item in eval_inputs]

    # Build the dataset object.
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    }
    
    try:
        dataset = Dataset.from_dict(data_dict)
        
        # Step 2: Call RAGAS evaluation.
        # We pass the dataset and specify the list of metrics.
        # Ragas will spawn OpenAI API queries to score each sample.
        scores_res = ragas_evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )
        
        # Convert response object to standard float dict.
        scores = {
            "faithfulness": float(scores_res.get("faithfulness", 0.0)),
            "answer_relevancy": float(scores_res.get("answer_relevancy", 0.0)),
            "context_precision": float(scores_res.get("context_precision", 0.0)),
            "context_recall": float(scores_res.get("context_recall", 0.0))
        }
        
        log.info("ragas_evaluation_complete", scores=scores)
        return scores
        
    except Exception as exc:
        log.error("ragas_evaluation_failed", error=str(exc))
        return None
