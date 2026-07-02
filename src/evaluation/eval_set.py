# src/evaluation/eval_set.py
# This module manages the loading and initialization of our evaluation dataset.
# The evaluation set contains questions, company ticker filters, and expected ground-truth answers.
# If the eval JSONL file does not exist, we automatically generate it with our 12 default queries
# to ensure the pipeline runs out-of-the-box.

from __future__ import annotations # Allow self-referencing type annotations.
import json # Standard library module to read and write JSON.
import os # Standard library module to manage file paths.
from typing import List # Type helper.
from src.shared.config import cfg # Config settings loader.
from src.shared.logging_setup import get_logger # Logger.
from src.shared.models import EvalQuery, QueryType # Shared models.

log = get_logger(__name__)

# List of 12 default seed evaluation queries across three difficulty levels.
DEFAULT_EVAL_QUERIES = [
    {
        "id": "Q01",
        "query": "What was Apple's total net revenue for fiscal year 2023, and how did it break down across product and services segments?",
        "query_type": "factual",
        "ticker": "AAPL",
        "filing_type": "10-K",
        "fiscal_year": 2023,
        "target_section": "Item 8",
        "expected_answer": "Apple's total net sales for fiscal year 2023 was $383,285 million, consisting of product sales of $298,085 million and services sales of $85,200 million."
    },
    {
        "id": "Q02",
        "query": "What was NVIDIA's research and development expense as a percentage of revenue in fiscal year 2024, and how does it compare to fiscal year 2023?",
        "query_type": "factual",
        "ticker": "NVDA",
        "filing_type": "10-K",
        "fiscal_year": 2024,
        "target_section": "Item 8",
        "expected_answer": "NVIDIA's R&D expense in fiscal year 2024 was $8,675 million (14.2% of revenue) compared to $7,339 million (27.2% of revenue) in fiscal year 2023."
    },
    {
        "id": "Q03",
        "query": "How many shares of common stock did Tesla repurchase in fiscal year 2023, and at what average price per share?",
        "query_type": "factual",
        "ticker": "TSLA",
        "filing_type": "10-K",
        "fiscal_year": 2023,
        "target_section": "Item 8",
        "expected_answer": "Tesla did not repurchase any shares of common stock in fiscal year 2023, as no repurchase program was active."
    },
    {
        "id": "Q04",
        "query": "What were JPMorgan Chase's net charge-offs as a percentage of average loans in Q3 2023, and what did management say drove the change versus Q3 2022?",
        "query_type": "factual",
        "ticker": "JPM",
        "filing_type": "10-Q",
        "fiscal_year": 2023,
        "target_section": "Item 2",
        "expected_answer": "Net charge-offs were 0.23% in Q3 2023. Management cited higher charge-offs in the Card Services segment as driving the increase from 0.12% in Q3 2022."
    },
    {
        "id": "Q05",
        "query": "What was Microsoft's operating income margin for fiscal year 2023 versus fiscal year 2022?",
        "query_type": "factual",
        "ticker": "MSFT",
        "filing_type": "10-K",
        "fiscal_year": 2023,
        "target_section": "Item 8",
        "expected_answer": "Microsoft's operating margin for FY23 was 41.8% ($88,523M operating income / $211,915M revenue) compared to 42.1% in FY22 ($83,383M income / $198,270M revenue)."
    },
    {
        "id": "Q06",
        "query": "Compare the gross profit margins of Apple, Microsoft, and Google (Alphabet) for their most recent fiscal years. Which company had the highest margin and why, according to their filings?",
        "query_type": "comparative",
        "ticker": None,
        "filing_type": "10-K",
        "fiscal_year": 2023,
        "target_section": "Item 8",
        "expected_answer": "Microsoft had the highest gross profit margin (68.9%) compared to Apple (44.1%) and Alphabet (56.5%) in fiscal year 2023. Microsoft cites its high-margin cloud infrastructure and software licenses."
    },
    {
        "id": "Q07",
        "query": "How do Tesla and Ford compare in terms of automotive gross margin for fiscal year 2023? What does each company cite as the primary driver of margin change?",
        "query_type": "comparative",
        "ticker": None,
        "filing_type": "10-K",
        "fiscal_year": 2023,
        "target_section": "Item 7",
        "expected_answer": "Tesla's automotive margin dropped to 19.4% in 2023 from 28.5% in 2022 due to vehicle price cuts. Ford's margin was lower, driven by legacy factory costs and electric division scaling issues."
    },
    {
        "id": "Q08",
        "query": "Which of JPMorgan Chase, Bank of America, and Goldman Sachs reported the highest return on equity (ROE) for fiscal year 2023, and which reported the lowest? What does each cite as the key lever for their ROE?",
        "query_type": "comparative",
        "ticker": None,
        "filing_type": "10-K",
        "fiscal_year": 2023,
        "target_section": "Item 7",
        "expected_answer": "Goldman Sachs reported the lowest ROE (7.5%), while JPMorgan Chase reported the highest ROE (17%) in fiscal year 2023. JPMorgan's lever was rising net interest income; Goldman was hit by real estate write-downs."
    },
    {
        "id": "Q09",
        "query": "Compare Amazon Web Services revenue growth rate to Microsoft Azure revenue growth rate for calendar year 2023. How does each company discuss the competitive dynamics between them?",
        "query_type": "comparative",
        "ticker": None,
        "filing_type": "10-K",
        "fiscal_year": 2023,
        "target_section": "Item 7",
        "expected_answer": "AWS grew 13% to $90.8B in 2023. Azure grew approximately 29-30% in growth rates. Both companies describe high competition and investments in AI capacity."
    },
    {
        "id": "Q10",
        "query": "Summarise the top five risk factors that NVIDIA identifies in its most recent 10-K. For each risk, quote the specific language from the filing that explains why NVIDIA considers it material.",
        "query_type": "qualitative",
        "ticker": "NVDA",
        "filing_type": "10-K",
        "fiscal_year": 2024,
        "target_section": "Item 1A",
        "expected_answer": "NVIDIA lists risks: 1. Concentration of customers, 2. Global trade and chip regulations, 3. Competition in GPU architectures, 4. Cybersecurity vulnerabilities, and 5. Production dependencies on external silicon foundries (like TSMC)."
    },
    {
        "id": "Q11",
        "query": "What cybersecurity risks does Microsoft disclose in its most recent 10-K, and what specific controls or mitigations does it describe?",
        "query_type": "qualitative",
        "ticker": "MSFT",
        "filing_type": "10-K",
        "fiscal_year": 2023,
        "target_section": "Item 1A",
        "expected_answer": "Microsoft identifies risks of state-sponsored cyberattacks targeting its cloud infrastructure. Mitigations include advanced zero-trust architectures and threat detection programs."
    },
    {
        "id": "Q12",
        "query": "How does Amazon describe its strategy for artificial intelligence and machine learning in its most recent 10-K and 10-Q filings? Has the language changed between the annual and quarterly reports?",
        "query_type": "qualitative",
        "ticker": "AMZN",
        "filing_type": "10-K",
        "fiscal_year": 2023,
        "target_section": "Item 1",
        "expected_answer": "Amazon focuses on integrating generative AI into AWS services and retail tools. The language shifted from general machine learning to LLM orchestration and custom silicon chips (Trainium, Inferentia)."
    }
]

def load_eval_set() -> List[EvalQuery]:
    """
    Loads the evaluation queries from data/eval/eval_set.jsonl.
    If the file does not exist, creates it with default queries first.
    """
    eval_file = cfg.evaluation.eval_set_path
    
    # Check if the file exists on the disk.
    if not os.path.exists(eval_file):
        log.info("eval_set_not_found", path=eval_file, action="creating_default_set")
        os.makedirs(os.path.dirname(eval_file), exist_ok=True)
        
        # Write the default queries to the JSONL file.
        with open(eval_file, "w", encoding="utf-8") as f:
            for query_dict in DEFAULT_EVAL_QUERIES:
                f.write(json.dumps(query_dict) + "\n")
                
    # Read and parse the evaluation queries.
    eval_queries: List[EvalQuery] = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                # Convert the JSON dictionary back to our structured Pydantic object.
                eval_queries.append(EvalQuery(**json.loads(line)))
                
    log.info("eval_set_loaded", path=eval_file, count=len(eval_queries))
    return eval_queries
