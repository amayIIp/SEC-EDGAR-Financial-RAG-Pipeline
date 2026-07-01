# Example User Queries — Seed Set for Evaluation

These 12 queries seed the evaluation harness built in Phase 7.
They span three difficulty tiers and three distinct query types.
`relevant_chunk_ids` and `expected_answer` fields are blank intentionally —
you will fill them in by running the ingestion/parsing/chunking pipeline
and manually verifying which chunk IDs contain the ground-truth information.

---

## Query Type Legend

| Type | What It Tests |
|------|---------------|
| **Factual** | Numeric precision, table-chunk retrieval, single-source grounding |
| **Comparative** | Multi-company retrieval, score attribution, cross-filing synthesis |
| **Qualitative** | Long-form summarisation, sub-topic segmentation, direct quotation |

---

## Factual Queries (Q01 – Q05)

### Q01 — Apple Segment Revenue (10-K 2023)
> **"What was Apple's total net revenue for fiscal year 2023, and how did it
> break down across product and services segments?"**

- **Why this query:** Tests numeric retrieval precision. The correct chunk must
  contain the consolidated income statement *or* the segment-disclosure note
  table. Validates that table chunks are stored with enough surrounding
  metadata that the income statement is surfaced ahead of unrelated revenue
  mentions elsewhere in the filing.
- **Failure mode to watch for:** The retriever surfaces an MD&A paragraph that
  *mentions* revenue directionally ("revenue declined slightly") but does not
  contain the actual figure — faithfulness score will be high but factual
  accuracy will be low.

---

### Q02 — NVIDIA R&D Expense Ratio, Two-Year Comparison (10-K 2024 vs 2023)
> **"What was NVIDIA's research and development expense as a percentage of
> revenue in fiscal year 2024, and how does it compare to fiscal year 2023?"**

- **Why this query:** Requires chunks from *two different annual filings* for
  the same company, plus a simple arithmetic derivation (R&D ÷ Revenue). Tests
  whether the date-range filter and multi-year corpus retrieval work correctly.
- **Failure mode to watch for:** The retriever returns only the FY2024 figure
  and the LLM fabricates the FY2023 comparison from parametric memory.

---

### Q03 — Tesla Share Repurchase Table (10-K 2023)
> **"How many shares of common stock did Tesla repurchase in fiscal year 2023,
> and at what average price per share?"**

- **Why this query:** The answer lives *exclusively* inside a structured HTML
  table in the equity notes section. Tests whether table chunks are indexed
  with sufficient surrounding context (caption, adjacent paragraph) to be
  retrieved by a natural-language query that contains none of the exact column
  headers.
- **Failure mode to watch for:** BM25 misses because the table rows contain
  numbers and terse labels with no query-matching vocabulary; vector search
  should compensate — making this a strong hybrid vs. BM25-only discriminator.

---

### Q04 — JPMorgan Net Charge-Off Rate with Management Commentary (10-Q Q3 2023)
> **"What were JPMorgan Chase's net charge-offs as a percentage of average
> loans in Q3 2023, and what did management say drove the change versus Q3
> 2022?"**

- **Why this query:** Two-part question requiring a quantitative metric *and*
  a nearby MD&A commentary paragraph. Tests whether the context-packing stage
  can combine a table chunk and a text chunk from the same section without
  one overwriting the other.
- **Failure mode to watch for:** The reranker surfaces only the table chunk
  (high numeric relevance) and drops the commentary paragraph, yielding an
  answer that answers part 1 but says "not found" for part 2.

---

### Q05 — Microsoft Operating Margin Year-over-Year (10-K 2023)
> **"What was Microsoft's operating income margin for fiscal year 2023 versus
> fiscal year 2022?"**

- **Why this query:** The filing reports raw operating income and revenue
  figures, not the derived margin percentage. The LLM must compute
  `operating_income / revenue × 100` from the retrieved numbers. Tests
  arithmetic reasoning capability across generator backends (GPT-4o-mini vs.
  Llama 3.1 8B) using the same retrieval context.
- **Failure mode to watch for:** The model uses the wrong year's figures
  (e.g. FY2021 from a comparative income statement), inflating apparent accuracy.

---

## Comparative Queries (Q06 – Q09)

### Q06 — Big-Tech Gross Margin 3-Way (10-K AAPL / MSFT / GOOGL)
> **"Compare the gross profit margins of Apple, Microsoft, and Google
> (Alphabet) for their most recent fiscal years. Which company had the highest
> margin and why, according to their filings?"**

- **Why this query:** Cross-company, cross-filing retrieval. Tests whether RRF
  fusion surfaces relevant results from all three company collections rather
  than over-indexing on the most common company in the corpus.
- **Failure mode to watch for:** The LLM correctly cites two companies but
  silently omits the third because no high-scoring chunk was retrieved for it —
  context precision drops but the answer appears complete at a glance.

---

### Q07 — Tesla vs. Ford Automotive Gross Margin (10-K 2023)
> **"How do Tesla and Ford compare in terms of automotive gross margin for
> fiscal year 2023? What does each company cite as the primary driver of
> margin change?"**

- **Why this query:** EV vs. legacy automaker. Tests correct chunk attribution —
  a very common failure mode is mixing up which margin figure belongs to which
  company when both appear in the same context window.
- **Failure mode to watch for:** The LLM uses Tesla's gross margin number when
  describing Ford or vice versa — subtle hallucination that looks plausible in
  isolation. Citation-checking utility should catch this.

---

### Q08 — Financial Sector ROE Ranking (10-K JPM / BAC / GS 2023)
> **"Which of JPMorgan Chase, Bank of America, and Goldman Sachs reported the
> highest return on equity (ROE) for fiscal year 2023, and which reported the
> lowest? What does each cite as the key lever for their ROE?"**

- **Why this query:** ROE is a derived metric (`net_income / avg_equity`).
  Tests whether the pipeline correctly retrieves the equity table, whether the
  LLM computes and correctly *ranks* three numeric values, and whether MD&A
  commentary for all three banks is retrieved and attributed.
- **Failure mode to watch for:** Hallucinated ranking (e.g. asserting Goldman
  had higher ROE than JPM when the data says otherwise).

---

### Q09 — AWS vs. Azure Cloud Revenue Growth (10-K AMZN / MSFT 2023)
> **"Compare Amazon Web Services revenue growth rate to Microsoft Azure
> revenue growth rate for calendar year 2023. How does each company discuss
> the competitive dynamics between them?"**

- **Why this query:** Tests asymmetric disclosure. AWS discloses absolute
  revenue (easy to compute growth); Azure discloses only a % growth figure
  (impossible to compute absolute size). Tests whether the LLM correctly
  handles disclosure asymmetry rather than fabricating an Azure revenue number.
- **Failure mode to watch for:** LLM estimates Azure's absolute revenue from
  parametric memory rather than faithfully reporting "Azure revenue in absolute
  dollars is not disclosed in the filing."

---

## Qualitative Queries (Q10 – Q12)

### Q10 — NVIDIA Top-5 Risk Factors with Direct Quotes (10-K 2024)
> **"Summarise the top five risk factors that NVIDIA identifies in its most
> recent 10-K. For each risk, quote the specific language from the filing that
> explains why NVIDIA considers it material."**

- **Why this query:** Long-form, multi-chunk summarisation over a dense
  multi-page Item 1A section. Tests whether structure-aware chunking correctly
  segments individual risk factors into independently retrievable chunks (rather
  than one giant 3000-token block), and whether the LLM synthesises across 5+
  chunks without losing source attribution.
- **Failure mode to watch for:** Quotes from one risk factor attributed to
  another; or fabricated quotes that paraphrase filing language rather than
  reproducing it verbatim.

---

### Q11 — Microsoft Cybersecurity Risk Disclosure (10-K 2023)
> **"What cybersecurity risks does Microsoft disclose in its most recent 10-K,
> and what specific controls or mitigations does it describe?"**

- **Why this query:** Tests sub-topic retrieval *within* a long section.
  Cybersecurity is a subsection of Item 1A. If the chunker merges all risk
  factors into one chunk, retrieval works fine for Q10 (all risk factors
  needed) but vector search may struggle to pinpoint the cybersecurity
  sub-chunk here, since there are many competing risk-factor chunks.
- **Failure mode to watch for:** Generic risk language retrieved instead of
  the specific cybersecurity subsection; no mitigation details cited.

---

### Q12 — Amazon AI Strategy, Annual vs. Quarterly Cross-Form Comparison
> **"How does Amazon describe its strategy for artificial intelligence and
> machine learning in its most recent 10-K and 10-Q filings? Has the language
> changed between the annual and quarterly reports?"**

- **Why this query:** Cross-form-type qualitative comparison. Tests metadata
  filter correctness (filing_type = "10-K" vs. "10-Q") and the LLM's ability
  to correctly attribute which statement came from which filing type. Also
  tests temporal sensitivity: if the Q has been filed after the K, does the
  pipeline surface the more recent language correctly?
- **Failure mode to watch for:** Conflating language across filings, or citing
  the 10-K's strategy section when asked about the 10-Q's update.

---

## Notes on Expanding to 50 Queries (Phase 7)

When building the full 50-query eval set, maintain approximately:

| Type | Count | Reasoning |
|------|-------|-----------|
| Factual | 20 | Most precise to grade; anchors retrieval metric (Recall@k) |
| Comparative | 15 | Stress-tests multi-company retrieval and attribution |
| Qualitative | 15 | Stress-tests generation quality and section segmentation |

For each query, after running the full pipeline:
1. Retrieve chunk IDs from the top-20 results.
2. Manually read each chunk and mark which ones actually contain information
   needed to answer the query — these become `relevant_chunk_ids`.
3. Write the `expected_answer` based on the filing text (not the model output).
4. Stage all three in `data/eval/eval_set.jsonl` before running Phase 7.
