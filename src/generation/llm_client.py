from __future__ import annotations 
import os 
from typing import List, Optional 
from openai import OpenAI 
try:
    from anthropic import Anthropic 
except ImportError:
    Anthropic = None 
import ollama 
import tiktoken 
from src.generation.citation_checker import check_citations 
from src.generation.prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE 
from src.shared.config import cfg 
from src.shared.logging_setup import get_logger 
from src.shared.models import CitedChunk, GenerationOutput, GenerationProvider 
log = get_logger(__name__)
class SECGenerator:
    """
    Coordinates context construction and LLM calls for OpenAI, Anthropic, or Ollama.
    """
    def __init__(self) -> None:
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_client = OpenAI(api_key=self.openai_key) if self.openai_key else None
        if Anthropic and self.anthropic_key:
            self.anthropic_client = Anthropic(api_key=self.anthropic_key)
        else:
            self.anthropic_client = None
        self.ollama_client = ollama.Client(host=cfg.generation.ollama.base_url)
        self.encoder = tiktoken.get_encoding(cfg.chunking.token_encoding)
    def _generate_openai(self, prompt: str, model: str) -> tuple[str, int, int]:
        """
        Sends the prompt to OpenAI API. Returns (answer_text, prompt_tokens, completion_tokens).
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=cfg.generation.openai.temperature,
            max_tokens=cfg.generation.openai.max_tokens,
            timeout=cfg.generation.openai.timeout
        )
        answer = response.choices[0].message.content or ""
        p_tokens = response.usage.prompt_tokens
        c_tokens = response.usage.completion_tokens
        return answer, p_tokens, c_tokens
    def _generate_anthropic(self, prompt: str, model: str) -> tuple[str, int, int]:
        """
        Sends the prompt to Anthropic API. Returns (answer_text, prompt_tokens, completion_tokens).
        """
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized. Install anthropic and set ANTHROPIC_API_KEY.")
        response = self.anthropic_client.messages.create(
            model=model,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=cfg.generation.anthropic.temperature,
            max_tokens=cfg.generation.anthropic.max_tokens,
            timeout=cfg.generation.anthropic.timeout
        )
        answer = response.content[0].text or ""
        p_tokens = response.usage.input_tokens
        c_tokens = response.usage.output_tokens
        return answer, p_tokens, c_tokens
    def _generate_ollama(self, prompt: str, model: str) -> tuple[str, int, int]:
        """
        Sends the prompt to local Ollama API. Returns (answer_text, prompt_tokens, completion_tokens).
        """
        fused_prompt = f"System: {SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"
        response = self.ollama_client.generate(
            model=model,
            prompt=fused_prompt,
            options={
                "temperature": cfg.generation.ollama.temperature,
                "num_predict": cfg.generation.ollama.num_predict
            }
        )
        answer = response.get("response", "")
        p_tokens = len(self.encoder.encode(fused_prompt))
        c_tokens = len(self.encoder.encode(answer))
        return answer, p_tokens, c_tokens
    def generate(
        self,
        query: str,
        chunks: List[CitedChunk],
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> GenerationOutput:
        """
        Orchestrates context prompt assembly, queries the chosen LLM backend,
        runs citation validation, and returns structured GenerationOutput.
        """
        llm_provider = (provider or cfg.generation.provider).lower()
        context_parts = []
        for chunk in chunks:
            context_parts.append(f"Context [{chunk.citation_tag}]:\n{chunk.packed_text}")
        context_text = "\n\n---\n\n".join(context_parts)
        user_prompt = USER_PROMPT_TEMPLATE.format(context_text=context_text, query=query)
        log.info("llm_generation_start", provider=llm_provider, num_chunks=len(chunks))
        if llm_provider == GenerationProvider.OPENAI.value:
            model_name = model or cfg.generation.openai.model
            answer, p_tokens, c_tokens = self._generate_openai(user_prompt, model_name)
        elif llm_provider == GenerationProvider.ANTHROPIC.value:
            model_name = model or cfg.generation.anthropic.model
            answer, p_tokens, c_tokens = self._generate_anthropic(user_prompt, model_name)
        elif llm_provider == GenerationProvider.OLLAMA.value:
            model_name = model or cfg.generation.ollama.model
            answer, p_tokens, c_tokens = self._generate_ollama(user_prompt, model_name)
        else:
            raise ValueError(f"Unknown generation provider: {llm_provider}")
        cited_tags, utilisation_ratio = check_citations(answer, chunks)
        log.info("llm_generation_complete", prompt_tokens=p_tokens, completion_tokens=c_tokens)
        return GenerationOutput(
            query=query,
            answer=answer,
            cited_tags=cited_tags,
            context_chunks=chunks,
            context_utilisation_ratio=utilisation_ratio,
            provider=GenerationProvider(llm_provider),
            model_name=model_name,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens
        )
