from __future__ import annotations 
import os 
import sys 
from unittest.mock import MagicMock 
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['cohere'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['ollama'] = MagicMock()
os.environ["OPENAI_API_KEY"] = "mock-openai-key-for-testing"
os.environ["COHERE_API_KEY"] = "mock-cohere-key-for-testing"
os.environ["ANTHROPIC_API_KEY"] = "mock-anthropic-key-for-testing"
os.environ["EDGAR_USER_AGENT"] = "TestAgent test@example.com"
