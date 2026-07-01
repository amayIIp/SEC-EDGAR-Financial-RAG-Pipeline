# tests/conftest.py
# This configuration file runs automatically before any test is loaded by pytest.
# We set up mock environment variables and mock heavy ML libraries (like PyTorch,
# SentenceTransformers, Cohere, OpenAI, and Ollama) globally.
# This prevents pytest from loading heavy deep learning weights during test collection,
# which eliminates Windows access violation crashes and makes tests run instantly.

from __future__ import annotations # Allow self-referencing type annotations.
import os # Standard library module for managing environment variables.
import sys # Standard library module to manipulate the Python runtime environment.
from unittest.mock import MagicMock # Standard library class for mock objects.

# ── Mocking heavy libraries globally to keep tests fast and CPU-only ──

# Mock sentence_transformers globally so we don't load local PyTorch models during test collection.
sys.modules['sentence_transformers'] = MagicMock()

# Mock the cohere client globally to avoid network connections or API client creation.
sys.modules['cohere'] = MagicMock()

# Mock the openai client globally to prevent outbound API client configuration.
sys.modules['openai'] = MagicMock()

# Mock the ollama client globally.
sys.modules['ollama'] = MagicMock()

# ── Mocking Environment Variables ──

# Set mock API keys for cloud services (OpenAI, Cohere, Anthropic).
# This ensures that our config loader does not raise validation errors when initializing.
os.environ["OPENAI_API_KEY"] = "mock-openai-key-for-testing"
os.environ["COHERE_API_KEY"] = "mock-cohere-key-for-testing"
os.environ["ANTHROPIC_API_KEY"] = "mock-anthropic-key-for-testing"

# Set a mock EDGAR user agent as required by the SEC client.
os.environ["EDGAR_USER_AGENT"] = "TestAgent test@example.com"
