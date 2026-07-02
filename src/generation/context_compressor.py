# src/generation/context_compressor.py
# This module implements context compression.
#
# =========================================================================================
# Advanced Concept: Context Compression
# When retrieving multiple large document chunks, we can hit context window limits
# or pay high API costs.
# Context compression addresses this by running each chunk through a fast, cheap LLM call
# (gpt-4o-mini) with instructions to summarize the text to ~40% of its length.
# The instruction enforces that all exact financial numbers, metrics, dates, and table formats
# are kept intact, while verbose explanations and filler words are removed.
# This yields highly dense, information-rich chunks that save token costs during the final QA step.
# =========================================================================================

from __future__ import annotations # Allow self-referencing type annotations.
import os # Standard library module to read environment variables.
from openai import OpenAI # Official OpenAI SDK client.
from src.generation.prompt_templates import COMPRESSION_PROMPT_TEMPLATE # Prompt template.
from src.shared.config import cfg # Config settings loader.
from src.shared.logging_setup import get_logger # Logger.

log = get_logger(__name__)

class ContextCompressor:
    """
    Compresses text chunks using a cheap, fast LLM call (gpt-4o-mini) to save context tokens.
    """

    def __init__(self) -> None:
        # Read the API key from environment variables.
        api_key = os.getenv("OPENAI_API_KEY")
        # Initialize client if key is present.
        self.client = OpenAI(api_key=api_key) if api_key else None

    def compress(self, text: str) -> str:
        """
        Compresses a text string using OpenAI to approximately 40% of its size.
        If the client is not initialized or compression is disabled, returns the text unmodified.
        """
        # Return unmodified text if client connection is missing.
        if not self.client:
            log.warning("compressor_not_initialized", action="returning_unmodified_text")
            return text

        # Format the compression prompt.
        prompt = COMPRESSION_PROMPT_TEMPLATE.format(text=text)

        try:
            # Call OpenAI chat completion endpoint.
            # We use temperature 0.0 to prevent the model from summarizing creatively.
            response = self.client.chat.completions.create(
                model=cfg.generation.compression_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=512, # Limit maximum tokens for output to save costs.
                timeout=15
            )
            
            # Extract and return the compressed text.
            compressed_text = response.choices[0].message.content or ""
            return compressed_text.strip()
            
        except Exception as exc:
            log.error("context_compression_failed", error=str(exc))
            # Fall back to returning original text if the compression call fails.
            return text
