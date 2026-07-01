# SEC EDGAR Financial RAG Pipeline

A production-grade, framework-free Retrieval-Augmented Generation (RAG) system built to query and analyze SEC HTML filings (10-K, 10-Q, 8-K) at scale. This project implements custom document parsing, structural table preservation, Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, and comprehensive performance profiling.

---

## 🏗️ Architecture & Pipeline Flow

Unlike standard RAG implementations that abstract execution behind orchestrators like LangChain or LlamaIndex, this system is built entirely using standard Python modules. This allows for deep performance optimization, synchronous stage profiling, and direct control over context construction.

```
                  ┌────────────────────────┐
                  │   SEC EDGAR Archives   │ (HTML filings, 10-K/Q, 8-K)
                  └───────────┬────────────┘
                              │ Ingest (Rate-Limited API Requests)
                              ▼
                  ┌────────────────────────┐
                  │  BeautifulSoup Parser  │ (Extracts text, converts tables to Markdown)
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Custom Section Chunker  │ (Regex-splits on SEC Item Headers)
                  └───────────┬────────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼ (Embedding Models)              ▼ (Lexical Indexing)
  ┌─────────────────────┐            ┌────────────────────┐
  │  OpenAI / BGE Large │            │    Elasticsearch   │
  └──────────┬──────────┘            └─────────┬──────────┘
             │ Vector Embeddings               │ BM25 Inverted Index
             ▼                                 ▼
  ┌─────────────────────┐            ┌────────────────────┐
  │     Qdrant DB       │            │   Elasticsearch    │
  └──────────┬──────────┘            └─────────┬──────────┘
             │ Semantic Hits                   │ Keyword Hits
             └────────────────┬────────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │  Reciprocal Rank Fusion │ (Combines ranked lists using RRF)
                  └───────────┬────────────┘
                              │ Fused Top Candidate Chunks
                              ▼
                  ┌────────────────────────┐
                  │  Cross-Encoder Rerank  │ (Cohere Rerank or Local BGE Reranker)
                  └───────────┬────────────┘
                              │ Top-N Most Relevant Context Chunks
                              ▼
                  ┌────────────────────────┐
                  │     LLM Generator      │ (GPT-4o-mini / Claude 3 / Local Llama 3.1)
                  └────────────────────────┘
```

### 1. Ingestion & Cleanup
*   **Source:** Direct, rate-limited requests to the SEC EDGAR API using designated User-Agents.
*   **Parsing:** Messy HTML is processed using `BeautifulSoup` and `lxml`. Instead of stripping table layouts, `<table>` tags are parsed and reconstructed as Markdown tables (`| Col 1 | Col 2 |`). This preserves cell relationships for search matching and LLM generation.

### 2. Custom Chunking Strategy
*   **Section-Awareness:** Splits documents dynamically by identifying SEC item headers (e.g., *Item 7. Management's Discussion and Analysis*, *Item 1A. Risk Factors*) using case-insensitive regular expressions.
*   **Recursive Splitter Fallback:** Within sections, blocks are merged and recursively split by structural separators (`\n\n`, `\n`, ` `) if they exceed `chunk_size` characters (default: 1500).
*   **Table Isolation:** Tables are treated as isolated chunks. If a table exceeds the threshold size, it is partitioned row-by-row while copying the table headers to each sub-table.

### 3. Dual Embedding Models (Experiment Setup)
*   **OpenAI `text-embedding-3-small`:** Generates 1536-dimensional vectors via API calls.
*   **BAAI `bge-large-en-v1.5`:** Generates 1024-dimensional vectors locally via the `sentence-transformers` library (supports GPU hardware acceleration).

### 4. Hybrid Search & Fusion
*   **Vector Search (Qdrant):** Performs fast cosine similarity searches. It supports exact CIK, year, and form type metadata filters.
*   **Lexical Search (Elasticsearch):** Performs BM25 keyword matching over chunk contents, optimized for numeric figures and exact terminology.
*   **Reciprocal Rank Fusion (RRF):** Merges vector and lexical results into a single ranked list. RRF scores items based on their relative ranks rather than raw match scores:
    $$\text{RRF Score} = \sum_{m \in \text{systems}} \frac{1}{60 + \text{Rank}_m}$$

### 5. Cross-Encoder Reranking
*   **Cohere Rerank API:** Utilizes the cloud-hosted `rerank-english-v3.0` model.
*   **Local BGE Reranker:** Executes the `bge-reranker-v2-m3` cross-encoder locally. Concatenates the query and candidate chunk, allowing the transformer self-attention layers to calculate a high-fidelity relevance grade.

### 6. Generation & Observability
*   **LLMs:** Supports OpenAI (GPT-4o-mini), Anthropic (Claude 3), and Local Ollama (Llama 3.1 8B).
*   **Observability:** Performance is profiled end-to-end. Latencies are recorded using high-precision CPU timers and visualized on a bar chart in the frontend.
*   **Evaluation:** Built-in automated checks calculate keyword recall on a hand-labeled golden dataset, alongside full automated `Ragas` LLM-as-a-judge evaluation.

---

## 🚀 Getting Started

### Prerequisites
*   [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed.
*   An OpenAI API key, Cohere API key, or Anthropic API key (optional, depending on the models you select).
*   If running Llama 3.1 locally: [Ollama](https://ollama.com/) running on your host system.

### Option A: Running with Docker Compose (Recommended)

1.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```bash
    # .env file configuration
    OPENAI_API_KEY=your-openai-api-key-here
    COHERE_API_KEY=your-cohere-api-key-here
    ANTHROPIC_API_KEY=your-anthropic-api-key-here
    ```

2.  **Start the Stack:**
    Run the following command to build the containers and launch Qdrant, Elasticsearch, FastAPI, and Streamlit:
    ```bash
    # Starts all services in the background and builds images
    docker compose up --build -d
    ```

3.  **Access the Applications:**
    *   **Streamlit Dashboard:** [http://localhost:8501](http://localhost:8501)
    *   **FastAPI Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

4.  **Shutdown:**
    ```bash
    # Stops and removes all running containers in the stack
    docker compose down
    ```

---

### Option B: Running Locally (Manual Setup)

If you prefer to run the databases in Docker but execute the API and Streamlit interfaces locally on your machine (recommended for local development and GPU usage):

1.  **Start Databases only:**
    ```bash
    # Start only the database containers
    docker compose up qdrant elasticsearch -d
    ```

2.  **Create a Virtual Environment & Install Dependencies:**
    ```bash
    # Create environment
    python -m venv venv
    # Activate environment (Windows)
    .\venv\Scripts\activate
    # Activate environment (Mac/Linux)
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    ```

3.  **Run FastAPI Backend:**
    ```bash
    # Run the server on port 8000
    uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
    ```

4.  **Run Streamlit Frontend:**
    ```bash
    # Start Streamlit UI on port 8501
    streamlit run src/streamlit_app.py --server.port 8501
    ```

---

## 🧪 Running System Evaluations

The pipeline includes a built-in evaluation suite that runs test queries against a hand-labeled golden dataset of SEC disclosures (e.g. Apple and Microsoft fiscal year 2023 figures).

1.  Go to the Streamlit UI ([http://localhost:8501](http://localhost:8501)).
2.  Select your desired pipeline configurations (Embedding Model, Reranker, Generator) in the sidebar.
3.  Click the **"Run Golden Set Evaluation"** button in the right-hand panel.
4.  The system will output:
    *   **Average Keyword Recall:** The percentage of critical numeric figures and financial terms correctly included in the answer.
    *   **Average Latency:** Total end-to-end execution time.
    *   **Section Matching Status:** Confirms whether the correct document segment (e.g. *Item 8* or *Item 1*) was fetched by the retriever.
    *   **Ragas Scores (Optional):** Grades for Faithfulness, Context Precision, and Recall when an OpenAI API key is supplied.
