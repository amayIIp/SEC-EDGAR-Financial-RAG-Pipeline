# src/generation/llm_client.py
# This module implements the Generation interface, supporting OpenAI,
# Anthropic, and local Ollama model backends.
# We pack retrieved context chunks into a unified prompt, send it to the LLM,
# and parse the response with citation checks and token usage counts.

from __future__ import annotations # Allow self-referencing type annotations.
import os # Standard library module to read environment variables.
from typing import List, Optional # Type helpers.
from openai import OpenAI # Official OpenAI SDK.
try:
    from anthropic import Anthropic # Official Anthropic SDK.
except ImportError:
    Anthropic = None # Handle optional import gracefully if missing.
import ollama # Local Ollama client library.
import tiktoken # Tokenizer to count prompt and completion tokens.
from src.generation.citation_checker import check_citations # Citation verification.
from src.generation.prompt_templates import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE # Prompt strings.
from src.shared.config import cfg # Config loader singleton.
from src.shared.logging_setup import get_logger # Logger.
from src.shared.models import CitedChunk, GenerationOutput, GenerationProvider # Shared models.

log = get_logger(__name__)

class SECGenerator:
    """
    Coordinates context construction and LLM calls for OpenAI, Anthropic, or Ollama.
    """

    def __init__(self) -> None:
        # Load API keys from environment.
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Initialize OpenAI client if key is present.
        self.openai_client = OpenAI(api_key=self.openai_key) if self.openai_key else None
        
        # Initialize Anthropic client if library is installed and key is present.
        if Anthropic and self.anthropic_key:
            self.anthropic_client = Anthropic(api_key=self.anthropic_key)
        else:
            self.anthropic_client = None

        # Initialize Ollama client pointing to local server URL.
        self.ollama_client = ollama.Client(host=cfg.generation.ollama.base_url)
        
        # Initialize token encoder for counting prompt/response tokens.
        self.encoder = tiktoken.get_encoding(cfg.chunking.token_encoding)

    def _generate_openai(self, prompt: str, model: str) -> tuple[str, int, int]:
        """
        Sends the prompt to OpenAI API. Returns (answer_text, prompt_tokens, completion_tokens).
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")

        # Send the chat completion request.
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
        
        # Extract response text and usage metadata.
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

        # Send the message request.
        # Anthropic passes the system prompt as a separate top-level parameter.
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
        
        # Extract text and usage tokens.
        answer = response.content[0].text or ""
        p_tokens = response.usage.input_tokens
        c_tokens = response.usage.output_tokens
        return answer, p_tokens, c_tokens

    def _generate_ollama(self, prompt: str, model: str) -> tuple[str, int, int]:
        """
        Sends the prompt to local Ollama API. Returns (answer_text, prompt_tokens, completion_tokens).
        """
        # Formulate a full single prompt combining system instructions.
        fused_prompt = f"System: {SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"
        
        # Call local generation.
        response = self.ollama_client.generate(
            model=model,
            prompt=fused_prompt,
            options={
                "temperature": cfg.generation.ollama.temperature,
                "num_predict": cfg.generation.ollama.num_predict
            }
        )
        
        answer = response.get("response", "")
        
        # Estimate token usage using our local encoder.
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
        
        # Step 1: Format context text blocks.
        context_parts = []
        for chunk in chunks:
            # We append the labeled text representation, e.g. "[1] Filing Source 1..."
            context_parts.append(f"Context [{chunk.citation_tag}]:\n{chunk.packed_text}")
        context_text = "\n\n---\n\n".join(context_parts)

        # Step 2: Format the user prompt template.
        user_prompt = USER_PROMPT_TEMPLATE.format(context_text=context_text, query=query)

        log.info("llm_generation_start", provider=llm_provider, num_chunks=len(chunks))

        # Step 3: Route query to the active model provider.
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

        # Step 4: Run citation checking to verify bracket citations.
        cited_tags, utilisation_ratio = check_citations(answer, chunks)

        log.info("llm_generation_complete", prompt_tokens=p_tokens, completion_tokens=c_tokens)

        # Step 5: Construct the final output model.
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
