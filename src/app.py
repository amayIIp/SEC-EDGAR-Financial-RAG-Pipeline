# src/app.py
# Root-level entrypoint that imports the main FastAPI app from the api package.
# This ensures backwards compatibility with any startup command pointed here.

from src.api.main import app
