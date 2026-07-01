# src/generation/prompt_templates.py
# This module defines the prompt templates used across our generation,
# context compression, and LLM-as-a-judge evaluation stages.
# Keeping prompts isolated makes it easy to iterate on system instructions.

from __future__ import annotations # Allow self-referencing type annotations.

# =============================================================================
# 1. RAG System Prompt
# Establishes the role, boundaries, and formatting rules for the LLM.
# Ordered to prevent hallucinations by forcing strict factual grounding.
# =============================================================================
SYSTEM_PROMPT = (
    "You are a professional, objective financial analyst database query assistant.\n"
    "Your task is to answer the user's question using ONLY the provided text and table chunks.\n"
    "Follow these constraints strictly:\n"
    "1. Cite your sources: for every fact, quote, or metric you write, append the matching "
    "citation tag (e.g. [1], [2]) at the end of the sentence.\n"
    "2. Ground your claims: do not make assumptions or extrapolate. If the context does not contain "
    "the information requested, state clearly: 'I am sorry, but the provided context does not contain "
    "the information required to answer this question.'\n"
    "3. Format tables in readable Markdown tables if requested or useful."
)

# =============================================================================
# 2. RAG User Prompt Template
# Wires the context blocks and the user query together.
# =============================================================================
USER_PROMPT_TEMPLATE = """Here is the retrieved context from SEC filings:

{context_text}

Query: {query}

Answer:"""

# =============================================================================
# 3. Context Compression Prompt
# Prompts a cheap LLM (gpt-4o-mini) to summarize a text chunk to ~40% of its length
# while retaining all financial numbers, entities, dates, and tables.
# =============================================================================
COMPRESSION_PROMPT_TEMPLATE = """You are a text compression engine.
Compress the following financial document chunk to approximately 40% of its original length.
CRITICAL: You must preserve all specific financial numbers, metrics, dates, percentages, and table formats.
Remove verbose explanations, adjectives, and filler words, but keep the core factual meaning intact.

Original Chunk:
\"\"\"
{text}
\"\"\"

Compressed Chunk:"""

# =============================================================================
# 4. LLM-as-a-Judge Evaluation Prompt
# Rubric to evaluate LLM responses on a scale from 1 to 5.
# Instructs the model to output a structured JSON containing a score and justification.
# =============================================================================
JUDGE_PROMPT_TEMPLATE = """You are an independent quality auditor evaluating a financial RAG system.
Evaluate the correctness and completeness of the generated answer compared to the ground-truth expected answer.

Question: {query}
Expected Answer (Ground Truth): {expected_answer}
Generated Answer to Evaluate: {answer}

Scoring Rubric:
- Score 5: The generated answer is fully correct, complete, and matches the expected answer perfectly.
- Score 4: The generated answer is correct and contains the key facts, but minor details are missing.
- Score 3: The generated answer is partially correct, but has omissions or minor inaccuracies.
- Score 2: The generated answer contains major inaccuracies or misses the main facts.
- Score 1: The generated answer is completely wrong, unsupported, or states it cannot answer when it should.

You must respond with a JSON object in this format:
{{
  "score": int,
  "justification": "Detailed explanation of the score based on the rubric."
}}

JSON Response:"""
