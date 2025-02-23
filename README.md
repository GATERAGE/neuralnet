# neuralnet
an exploration of RAGE training integration with a mini transformer<br />
thanks to the team at Stanford for the work on alpaca<br />
```bash
ollama serve
ollama list
ollama run deepseek-r1:1.5b

```bash
neuralnet/
 ├─ DOC.md
 ├─ server.js
 ├─ requirements.txt
 ├─ index.html
 ├─ style.css
 ├─ memory/                (auto-created to store ingest/inference logs)
 ├─ docs/                  (where user-pasted text is saved)
 ├─ rag_inference.py
 ├─ rage_dataloader.py
 ├─ llm_router.py
 ├─ production_transformer.py

```
# RAGE Project Documentation

## Overview

This project implements a **Retrieval Augmented Generative Engine as template** pipeline, combining:
1. **Node.js server** for ingestion and inference endpoints.
2. **Python scripts** that handle chunking, embedding, indexing (FAISS), and LLM backends (local, OpenAI, Together.ai, Ollama).

The ingestion results and final inference outputs are saved in a `memory/` folder for easy auditing/debugging.

# Table of Contents

    Introduction
    Project Goals
    Architecture Overview
    Data Ingestion & Chunking
    RAGE Transformer Components
        production_transformer
        rage_dataloader
        llm_router
        rag_inference
    UI & User Experience
    Best Practices for Indexing & Retrieval
    Deployment & Environment Variables
    Summary & Next Steps

# Introduction

Retrieval-Augmented Generative Engine (RAGE) is a dynamic data bridge for large language models (LLMs) and external knowledge sources. By retrieving relevant text chunks from a vector index (e.g., FAISS) and augmenting the user query with this context, RAG-based systems can produce more grounded and up-to-date responses.

The RAGE Transformer Project is an end-to-end implementation of a RAGE pipeline. It focuses on multiple data types (TXT, MD, PDF, DOCX, remote URLs), a minimal local Transformer (ProductionTransformer), and integration with external LLM APIs (OpenAI, Together.ai, Ollama). A Node.js server provides a straightforward UI to ingest data and query the pipeline.
Project Goals

    Modular Data Ingestion: Handle local folders, file uploads, and remote URLs, chunkifying text for efficient retrieval.
    Unified Index: Use FAISS to build and store embeddings, enabling top-k retrieval.
    Pluggable LLMs: Offer a simple router (LLMRouter) to switch between local or API-based inference.
    Comprehensive UI: Provide a pure Node.js front-end for Data Ingestion and Query Inference.
    Best Practices: Keep secrets out of code, maintain a robust folder structure, and follow naming conventions for clarity.

# Architecture Overview

The RAGE Transformer system comprises:

    rage_dataloader: Loads and chunkifies data from multiple sources and file types.
    FAISS Index: Stores text embeddings computed by SentenceTransformers.
    rag_inference: Coordinates ingestion, indexing, and retrieval. Merges user query with retrieved chunks.
    llm_router: Chooses which LLM backend to call (local transformer, OpenAI, Together.ai, or Ollama).
    production_transformer: A minimal PyTorch Transformer that can serve as a local LLM.
    Node.js front-end: Serves index.html and style.css; handles ingestion and inference requests.

[User Query] --> [Node.js] --> [rag_inference.py] --> [FAISS: Retrieve Chunks] 
                --> [LLMRouter: Local/OpenAI/Together/Ollama] --> [Final Response]

# Data Ingestion & Chunking
Multiple Data Formats

    TXT / Markdown: Basic text files are read linearly and chunked by words.
    PDF: Parsed via PyPDF2, extracting textual data from pages.
    DOCX: Read with python-docx, extracting paragraph text.
    URLs: Fetched as raw text via requests; can be HTML or plain text.

# Chunking Strategy

In rage_dataloader.py, every file is split into chunks of ~128 words (configurable). Word-level chunking ensures relevant segments remain contextually intact, while preventing excessively large vectors.

for i in range(0, len(words), self.chunk_size):
    chunk = " ".join(words[i:i+self.chunk_size])
    chunks.append(chunk)

Why chunk? It gives the FAISS index better granularity for retrieval. Instead of indexing entire documents, we store smaller text pieces—making results more targeted.
RAGE Transformer Components
ProductionTransformer

    Module: production_transformer.py
    Role: Minimal PyTorch encoder that can function as a local language model (placeholder).
    Naming Convention: Prefixed with “Production” to highlight it’s a building block ready for deployment.
    Key Methods:
        forward(x, mask=None): Takes token IDs and optional causal mask, returns logits.

# rage_dataloader

    Module: rage_dataloader.py
    Role: Collect, parse, chunk data from local or remote sources.
    Naming Convention: “RAGE” stands for Retrieval-Augmented Generative Engine, focusing on the ingestion side.
    Features:
        _load_pdf_file, _load_docx_file, _load_text_file: Specialized loaders for each format.
        _chunk_text: Splits text into word-based chunks, e.g. 128 words each.

# llm_router

    Module: llm_router.py
    Role: Takes the final augmented prompt and dispatches it to a desired LLM backend.
    Naming Convention: “Router” clarifies it’s not generating text by itself but routing the prompt to a local or remote model.
    Backends:
        local (ProductionTransformer)
        openai (requires OPENAI_API_KEY)
        together (requires TOGETHER_API_KEY)
        ollama (calls a local Llama-based server via an HTTP endpoint)

# rag_inference

    Module: rag_inference.py
    Role: Ties ingestion, FAISS indexing, and inference together.
    Naming Convention: “RAGInference” clearly signals a Retrieval-Augmented Generation process.
    Key Routines:
        build_or_load_index(): If existing FAISS index files are found, load them. Otherwise, build from the default docs/ folder.
        ingest_data(): Merges newly provided data (files, folders, URLs) with the existing index.
        retrieve_context(query, top_k=3): Returns top-k chunk matches.
        generate_response(query): Merges user query and retrieved context, then calls LLMRouter.

# UI & User Experience

Node.js serves a minimal front-end (index.html + style.css) that:

    Data Ingestion Form:
        Folder Path: Points to local directories with PDF, DOCX, TXT/MD files.
        URL: Fetches remote text or HTML.
        Pasted Content: Allows direct input of arbitrary text.
    Query Form:
        Query: A user question or prompt (e.g., “Summarize our Q4 strategy.”).
        LLM Backend: Choose from local, OpenAI, Together, or Ollama.

By separating ingestion and inference steps, the user can gradually build up the FAISS index from multiple sources. They can then query the combined knowledge base.
Best Practices for Indexing & Retrieval

    Granular Chunks: Aim for chunk sizes of 256–512 tokens (or ~128–300 words). This balance helps retrieval systems find just enough context.
    Regular Index Updates: If new documents arrive or older ones get replaced, rebuild the index or adopt partial indexing methods.
    Metadata Tracking: (Optional) Store metadata (e.g., source file name, creation date) alongside chunks.
    Vector Normalization: Most Sentence Transformers produce normalized embeddings. For FAISS IndexFlatIP, ensure embeddings are float32 and in consistent shape (N, d).
    Masking & Tokenization (for local model): The ProductionTransformer code includes a basic positional encoding but doesn’t show tokenization or generation loops. In production, integrate a real tokenizer (e.g., HuggingFace) for the local model.

# Deployment & Environment Variables

Secrets:

    OPENAI_API_KEY: for OpenAI GPT usage.
    TOGETHER_API_KEY: for Together.ai.
    OLLAMA_ENDPOINT: default http://localhost:11411.

Store them in environment variables or a .env file—never commit keys. For example:

# .env (do not commit)
OPENAI_API_KEY="sk-..."
TOGETHER_API_KEY="tog-..."
OLLAMA_ENDPOINT="http://localhost:11411"

# Running the Project:

    Install Python dependencies (torch, faiss-cpu, sentence-transformers, PyPDF2, python-docx).
    Node.js environment to run server.js.
    Start the server:

```bash
export PYTHON_PATH="/full/path/to/venv/bin/python"
node server.js
```

    RAGE Transformer is a cohesive retrieval-augmented generation system:
        Multi-format ingestion (PDF, DOCX, TXT, URLs)
        FAISS-based vector retrieval
        Local or Remote LLM via a pluggable router
    The ProductionTransformer demonstrates a minimal PyTorch architecture, which can be extended into a more sophisticated local LLM or replaced entirely.
    The Node.js UI showcases straightforward ingestion and query forms, easily extensible with more advanced front-end features (drag-and-drop file uploads, user authentication, etc.).

# Potential Enhancements

    Automatic chunk overlap: Provide some overlap between chunks to capture partial context boundaries.
    Better PDF parsing: Handle scanned PDFs (OCR), advanced layout (tables).
    Improve Local Generation: Integrate advanced decoding strategies (beam search, top-k, nucleus sampling) for the local model.
    Metadata Search: Alongside text retrieval, store doc metadata to refine or filter results.
    Distributed Indexing: For very large corpora, consider a distributed FAISS or a specialized system (e.g., Milvus).

# Chunking Strategy

In rage_dataloader.py, every file is split into chunks of ~128 words (configurable). Word-level chunking ensures relevant segments remain contextually intact, while preventing excessively large vectors.

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

## Setup
```bash
git clone https://github.com/GATERAGE/neuralnet
cd neuralnet
python3.10 -m venv rage
source venv/bin/activate  # or venv\Scripts\activate on Windows
python --version
pip install torch torchvision torchaudio
pip install faiss-cpu          # or faiss-gpu if GPU available
pip install sentence-transformers request
pip install openai             # required as checked before using ollama model
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

# Usage

    Ingest Data: Provide chunk size, folder path (./docs), URLs, or paste text. Click Ingest.
    Query: Enter a query, pick LLM backend, click Submit.
    Result: The system retrieves top-k chunks from FAISS, merges them into a prompt, and calls the chosen LLM.

# Environment Variables

    OPENAI_API_KEY
    TOGETHER_API_KEY
    OLLAMA_ENDPOINT (default http://localhost:11411)


   - Provide a folder path like `./docs` or a remote URL.  
   - Hit **Ingest**.  
   - Then enter a query (“Summarize the doc...”), pick LLM backend (local/OpenAI/etc.), and click **Submit**.

You now have a **production-oriented** RAGE system that can dynamically adapt chunk size for large context windows. Open Source or go away.
