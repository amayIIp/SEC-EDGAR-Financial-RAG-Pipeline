from __future__ import annotations 
import os 
from openai import OpenAI 
from src.generation.prompt_templates import COMPRESSION_PROMPT_TEMPLATE 
from src.shared.config import cfg 
from src.shared.logging_setup import get_logger 
log = get_logger(__name__)
class ContextCompressor:
    """
    Compresses text chunks using a cheap, fast LLM call (gpt-4o-mini) to save context tokens.
    """
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
    def compress(self, text: str) -> str:
        """
        Compresses a text string using OpenAI to approximately 40% of its size.
        If the client is not initialized or compression is disabled, returns the text unmodified.
        """
        if not self.client:
            log.warning("compressor_not_initialized", action="returning_unmodified_text")
            return text
        prompt = COMPRESSION_PROMPT_TEMPLATE.format(text=text)
        try:
            response = self.client.chat.completions.create(
                model=cfg.generation.compression_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=512, 
                timeout=15
            )
            compressed_text = response.choices[0].message.content or ""
            return compressed_text.strip()
        except Exception as exc:
            log.error("context_compression_failed", error=str(exc))
            return text
