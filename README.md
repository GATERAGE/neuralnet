# neuralnet
an exploration of RAGE integration with a mini transformer

```bash
my_project/
 ├─ production_transformer.py
 ├─ rage_dataloader.py
 ├─ llm_router.py
 ├─ rag_inference.py
 ├─ server.js
 ├─ index.html
 ├─ style.css
 ├─ README.md
 ├─ INSTALL.md
 └─ docs/                  (default folder for local files)
```
# Introduction

Retrieval-Augmented Generation (RAG) is a cutting-edge technique that bridges large language models (LLMs) and external knowledge sources. By retrieving relevant text chunks from a vector index (e.g., FAISS) and augmenting the user query with this context, RAG-based systems can produce more grounded and up-to-date responses.

The RAGE Transformer Project is an end-to-end implementation of a RAG pipeline. It focuses on multiple data types (TXT, MD, PDF, DOCX, remote URLs), a minimal local Transformer (ProductionTransformer), and integration with external LLM APIs (OpenAI, Together.ai, Ollama). A Node.js server provides a straightforward UI to ingest data and query the pipeline.
Project Goals

    Modular Data Ingestion: Handle local folders, file uploads, and remote URLs, chunkifying text for efficient retrieval.
    Unified Index: Use FAISS to build and store embeddings, enabling top-k retrieval.
    Pluggable LLMs: Offer a simple router (LLMRouter) to switch between local or API-based inference.
    Comprehensive UI: Provide a pure Node.js front-end for Data Ingestion and Query Inference.
    Best Practices: Keep secrets out of code, maintain a robust folder structure, and follow naming conventions for clarity.

Architecture Overview

The RAGE Transformer system comprises:

    RAGEDataLoader: Loads and chunkifies data from multiple sources and file types.
    FAISS Index: Stores text embeddings computed by SentenceTransformers.
    RAGInference: Coordinates ingestion, indexing, and retrieval. Merges user query with retrieved chunks.
    LLMRouter: Chooses which LLM backend to call (local transformer, OpenAI, Together.ai, or Ollama).
    ProductionTransformer: A minimal PyTorch Transformer that can serve as a local LLM.
    Node.js front-end: Serves index.html and style.css; handles ingestion and inference requests.

    ```bash
    [User Query] --> [Node.js] --> [rag_inference.py] --> [FAISS: Retrieve Chunks] 
                --> [LLMRouter: Local/OpenAI/Together/Ollama] --> [Final Response]
    ```



# Retrieval-Augmented Generation with Configurable Chunk Size

This project provides a multi-format data ingestion pipeline (PDF, DOCX, TXT, MD, URLs), chunking them at a user-configurable size (e.g. 128 words, 4096 words, etc.), storing embeddings in FAISS, then augmenting user queries with retrieved chunks. The final prompt is routed to a local minimal Transformer or external LLM API (OpenAI, Together.ai, Ollama)


---

## Running & Testing

1. **Install** all Python deps and confirm Node version.  
2. **`node server.js`**  
3. **Open** [http://localhost:3000](http://localhost:3000):
   - In **Data Ingestion**, set **Chunk Size** (e.g. 4096).  
   - Provide a folder path like `./docs` or a remote URL.  
   - Hit **Ingest**.  
   - Then enter a query (“Summarize the doc...”), pick LLM backend (local/OpenAI/etc.), and click **Submit**.

You now have a **production-oriented** RAGE system that can dynamically adapt chunk size for large context windows. Enjoy!


## Setup
```bash
git clone https://github.com/GATERAGE/neuralnet
cd neuralnet
python -m venv rage
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install torch torchvision torchaudio
pip install faiss-cpu          # or faiss-gpu if GPU available
pip install sentence-transformers requests
pip install PyPDF2             # optional, for PDF parsing
pip install python-docx        # optional, for DOCX parsing
```

   ```bash
   node --version   # should be >= 14
   ```
   ```bash
   node server.js
   ```
   Open http://localhost:3000 in your browser

   Usage

    Ingest Data: Provide chunk size, folder path (./docs), URLs, or paste text. Click Ingest.
    Query: Enter a query, pick LLM backend, click Submit.
    Result: The system retrieves top-k chunks from FAISS, merges them into a prompt, and calls the chosen LLM.

Environment Variables

    OPENAI_API_KEY
    TOGETHER_API_KEY
    OLLAMA_ENDPOINT (default http://localhost:11411)


 How to Use the Multi-Source RAG Project
 
# Project Overview

This project sets up an end-to-end Retrieval-Augmented Generation pipeline that can:

    Ingest files from local folders (TXT, Markdown, PDF, DOCX) or remote URLs.
    Chunk text into manageable segments (~128 words each).
    Embed chunks using a SentenceTransformer model and store them in a FAISS index.
    Retrieve top-k relevant chunks for any user query.
    Augment the user’s query with retrieved context, then generate a final response using either:
        A local minimal PyTorch Transformer (ProductionTransformer), or
        An external LLM API (OpenAI, Together.ai, or a local Ollama server).

A Node.js server hosts a simple web front-end (index.html) for ingestion and query submission

# Prerequisites

    Python 3.7+
    Node.js 14+
    PyTorch installed (pip install torch torchvision torchaudio)
    faiss-cpu (or faiss-gpu if you have a CUDA-compatible GPU)
    sentence-transformers
    PyPDF2 (for PDF ingestion) and python-docx (for DOCX ingestion), if you need those file types.

---

## Running & Testing

1. **Install** all Python deps and confirm Node version.  
2. **`node server.js`**  
3. **Open** [http://localhost:3000](http://localhost:3000):
   - In **Data Ingestion**, set **Chunk Size** (e.g. 4096).  
   - Provide a folder path like `./docs` or a remote URL.  
   - Hit **Ingest**.  
   - Then enter a query (“Summarize the doc...”), pick LLM backend (local/OpenAI/etc.), and click **Submit**.

You now have a **production-oriented** RAG system that can dynamically adapt chunk size for large context windows. Enjoy!

   

