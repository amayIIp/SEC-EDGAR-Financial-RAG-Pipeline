from __future__ import annotations 
import json 
import os 
from typing import Any, Dict, Optional 
from openai import OpenAI 
from src.generation.prompt_templates import JUDGE_PROMPT_TEMPLATE 
from src.shared.config import cfg 
from src.shared.logging_setup import get_logger 
log = get_logger(__name__)
class LLMJudge:
    """
    Grades generated answers against ground-truth expected answers using an LLM auditor.
    """
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
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
        if not self.client:
            log.warning("llm_judge_not_initialized", action="skipping_grade")
            return {"score": 1, "justification": "LLM Judge client not configured. Set OPENAI_API_KEY."}
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=query,
            expected_answer=expected_answer,
            answer=answer
        )
        try:
            response = self.client.chat.completions.create(
                model=cfg.evaluation.judge.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=cfg.evaluation.judge.temperature,
                response_format={"type": "json_object"},
                max_tokens=256,
                timeout=20
            )
            raw_text = response.choices[0].message.content or "{}"
            result = json.loads(raw_text)
            score = int(result.get("score", 1))
            justification = str(result.get("justification", "No justification provided."))
            log.info("llm_judge_scored_sample", query=query[:40], score=score)
            return {"score": score, "justification": justification}
        except Exception as exc:
            log.error("llm_judge_failed", error=str(exc))
            return {"score": 1, "justification": f"LLM Judge scoring failed: {exc}"}
