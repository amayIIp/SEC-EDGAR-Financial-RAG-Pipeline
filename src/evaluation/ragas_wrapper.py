from __future__ import annotations 
import os 
from typing import Any, Dict, List, Optional 
from src.shared.config import cfg 
from src.shared.logging_setup import get_logger 
log = get_logger(__name__)
try:
    from ragas import evaluate as ragas_evaluate 
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall 
    from datasets import Dataset 
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
    if not RAGAS_AVAILABLE:
        log.warning("ragas_library_unavailable", action="skipping_ragas_eval")
        return None
    if not os.getenv("OPENAI_API_KEY"):
        log.warning("openai_key_missing_for_ragas", action="skipping_ragas_eval")
        return None
    log.info("starting_ragas_evaluation", count=len(eval_inputs))
    questions = [item["question"] for item in eval_inputs]
    answers = [item["answer"] for item in eval_inputs]
    contexts = [item["contexts"] for item in eval_inputs]
    ground_truths = [[item["ground_truth"]] for item in eval_inputs]
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    }
    try:
        dataset = Dataset.from_dict(data_dict)
        scores_res = ragas_evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )
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
