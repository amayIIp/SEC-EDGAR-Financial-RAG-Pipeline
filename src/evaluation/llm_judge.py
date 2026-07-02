# src/evaluation/llm_judge.py
# This module implements the LLM-as-a-judge correctness scorer.
#
# =========================================================================================
# Advanced Concept: LLM-as-a-Judge Correctness Scoring
# Standard text matching metrics (like BLEU or ROUGE) measure exact n-gram overlap.
# They are notoriously poor for financial answers, where changing a single word (e.g., "declined"
# to "increased") changes the correctness entirely, while paraphrase answers are graded poorly.
# We implement a semantic correctness auditor: we prompt a high-performing LLM (gpt-4o-mini)
# with the Question, the Expected Answer, and the Generated Answer.
# It evaluates semantic equivalence and completeness using a strict 1-5 scoring rubric,
# returning a structured JSON object containing a numerical grade and textual justification.
# =========================================================================================

from __future__ import annotations # Allow self-referencing type annotations.
import json # Standard library module to parse JSON.
import os # Standard library module to read environment variables.
from typing import Any, Dict, Optional # Type helpers.
from openai import OpenAI # Official OpenAI SDK client.
from src.generation.prompt_templates import JUDGE_PROMPT_TEMPLATE # Prompt template.
from src.shared.config import cfg # Config settings loader.
from src.shared.logging_setup import get_logger # Logger.

log = get_logger(__name__)

class LLMJudge:
    """
    Grades generated answers against ground-truth expected answers using an LLM auditor.
    """

    def __init__(self) -> None:
        # Load API key.
        api_key = os.getenv("OPENAI_API_KEY")
        # Initialize client if key is present.
        self.client = OpenAI(api_key=api_key) if api_key else None

    def grade_answer(
        self,
        query: str,
        answer: str,
        expected_answer: str
    ) -> Dict[str, Any]:
        """
        Queries OpenAI to grade a generated answer.
        Returns a dict: {"score": int, "justification": str}
        """
        # Return default failure result if client is not configured.
        if not self.client:
            log.warning("llm_judge_not_initialized", action="skipping_grade")
            return {"score": 1, "justification": "LLM Judge client not configured. Set OPENAI_API_KEY."}

        # Format the judge audit prompt.
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=query,
            expected_answer=expected_answer,
            answer=answer
        )

        try:
            # Call OpenAI chat completion endpoint.
            # We enforce JSON response format if supported, or extract it from text.
            response = self.client.chat.completions.create(
                model=cfg.evaluation.judge.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=cfg.evaluation.judge.temperature,
                # Enforce JSON output for structured parsing.
                response_format={"type": "json_object"},
                max_tokens=256,
                timeout=20
            )
            
            # Parse the response text.
            raw_text = response.choices[0].message.content or "{}"
            result = json.loads(raw_text)
            
            # Verify parsed keys exist.
            score = int(result.get("score", 1))
            justification = str(result.get("justification", "No justification provided."))
            
            log.info("llm_judge_scored_sample", query=query[:40], score=score)
            return {"score": score, "justification": justification}
            
        except Exception as exc:
            log.error("llm_judge_failed", error=str(exc))
            # Return fallback on error.
            return {"score": 1, "justification": f"LLM Judge scoring failed: {exc}"}

