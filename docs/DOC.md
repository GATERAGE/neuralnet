# neuralnet
An exploration of **RAGE** (Retrieval‑Augmented Generative Engine) training integration with a mini Transformer.

> **RAGE** = Retrieval‑Augmented Generative Engine  
> **GATE** = General Automation Technology Environment  
> **Website:** https://rage.pythai.net

This repository provides a runnable **RAG template pipeline**:
- **Node.js server + UI** for ingestion and inference
- **Python pipeline** for chunking, embeddings, FAISS retrieval, and LLM routing
- **Memory logs** saved to `memory/` for auditability (“logs are memory”)

---

## Quickstart

### 1) Prerequisites
- **Node.js** ≥ 16 (18+ recommended)
- **Python** ≥ 3.10 (3.11 recommended)

### 2) Create a Python environment
```bash
git clone https://github.com/GATERAGE/neuralnet
cd neuralnet

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

python --version
pip install --upgrade pip
```

### 3) Install Python dependencies
Recommended baseline (CPU):
```bash
pip install torch numpy requests
pip install faiss-cpu
pip install sentence-transformers
pip install transformers
```

Optional format support:
```bash
pip install pypdf python-docx
```

Optional LLM API clients:
```bash
pip install openai
```

> If you plan to use a GPU, install the correct PyTorch build for your platform first, then use `faiss-gpu` if appropriate.

### 4) Start the Node server
```bash
export PYTHON_PATH="$PWD/.venv/bin/python"
node server.js
```

Open:
- http://localhost:3000

---

## Ollama (local LLM backend)

If you want to route inference to a local Ollama model:

```bash
ollama serve
ollama list
ollama run deepseek-r1:1.5b
```

Set the endpoint if your router supports it (default varies by your setup):
```bash
export OLLAMA_ENDPOINT="http://localhost:11411"
```

---

## Repository layout

```text
neuralnet/
 ├─ DOC.md
 ├─ TECHNICAL.md
 ├─ USAGE.md
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
 └─ (optional) tools/      (ipfs_fetch.py, generate.py, etc.)
```

---

## Overview

RAGE bridges LLMs and external knowledge sources by:
1) **Ingesting** multi-format data (TXT, MD, PDF, DOCX, URLs)
2) **Chunking** content into retrieval units
3) **Embedding** chunks with SentenceTransformers
4) **Indexing** embeddings in FAISS for top‑k similarity search
5) **Augmenting** the user query with retrieved chunks
6) **Routing** the augmented prompt to a backend model (local/API/Ollama)

**Execution flow**
```text
[User Query] → [Node.js UI/API] → [rag_inference.py]
          → [FAISS Retrieve Chunks] → [LLM Router]
          → [Final Response + memory logs]
```

---

## Project goals

- **Modular ingestion:** folders, URLs, pasted text; extensible to uploads
- **Unified index:** persistent FAISS index with reproducible chunking
- **Pluggable generation:** local transformer scaffold + remote APIs + Ollama
- **Auditability:** ingestion/inference artifacts written to `memory/`
- **Security hygiene:** secrets in environment variables, not in code

---

## Components

### `rage_dataloader.py` — multi-format loader + chunker
- Loads `.txt`, `.md`, `.pdf`, `.docx` and URL text
- Chunking is **word-based** by default (configurable, e.g. 128 words)

**Why chunking matters**
- Improves retrieval granularity
- Reduces irrelevant context
- Controls index size and latency

> Note: word-based chunking is a prototype default. For production, consider token-based chunking with overlap.

---

### `rag_inference.py` — RAG orchestration
Responsible for:
- building or loading FAISS index
- ingesting new sources
- retrieving top‑k relevant chunks
- constructing the augmented prompt
- calling `llm_router.py`

---

### `llm_router.py` — backend selector
Routes prompts to:
- **local** (minimal transformer scaffold)
- **OpenAI** (requires `OPENAI_API_KEY`)
- **Together** (requires `TOGETHER_API_KEY`)
- **Ollama** (local HTTP endpoint)

---

### `production_transformer.py` — minimal local transformer
A minimal PyTorch Transformer used as a local scaffold for logits and wiring.
It is not intended to compete with full production LLM stacks by itself.

**Production note**
- A real local model requires a tokenizer + decoding loop + caching.
- In production, you may replace this with a model server/runtime while keeping RAGE retrieval constant.

---

## UI & user experience

The Node server serves:
- `index.html` + `style.css` for a minimal UI
- endpoints for:
  - ingestion (folder path / URL / pasted text)
  - inference (query + backend selection)

This separation enables incremental indexing and repeated queries over the same knowledge base.

---

## Best practices for indexing & retrieval

- **Chunk sizing:** aim for ~256–512 tokens (or ~128–300 words) depending on content
- **Chunk overlap:** improves boundary recall for long documents
- **Metadata:** store `{source, filename, timestamp, integrity tier}` alongside chunks
- **Hybrid retrieval:** combine lexical (BM25) + vector + reranking for best results
- **Normalization:** ensure embeddings are float32 and use a consistent similarity metric

---

## Environment variables

Never commit secrets.

Common variables:
```bash
OPENAI_API_KEY="sk-..."
TOGETHER_API_KEY="tog-..."
OLLAMA_ENDPOINT="http://localhost:11411"
PYTHON_PATH="/path/to/.venv/bin/python"
```

---

## Running & testing

1) Install deps + confirm Node:
```bash
node --version
python --version
```

2) Start server:
```bash
export PYTHON_PATH="$PWD/.venv/bin/python"
node server.js
```

3) Open UI:
- http://localhost:3000

4) Ingest data:
- set chunk size
- provide folder path / URL / paste text
- click **Ingest**

5) Query:
- enter question
- choose backend (local / OpenAI / Together / Ollama)
- click **Submit**

---

## Known limitations

### PDF parsing
Many PDFs are not “text PDFs” (scanned, complex layout). Text extraction may return empty or poor output.

Recommended upgrades:
- use PyMuPDF as primary extractor, pypdf fallback
- OCR only when needed

### Local generation
The minimal transformer is a scaffold:
- no tokenizer
- no strong decoding strategies
- not optimized for long context

Use Ollama or an API backend for real generation until a local runtime is integrated.

### Prototype security posture
This repo is a template and should not be exposed publicly as‑is:
- validate inputs (especially folder paths)
- add authentication/rate limits
- sandbox spawned processes

---

## Potential enhancements

- Token-based chunking + overlap windows
- Hybrid retrieval + reranking (e.g., **SimpleMind** as policy/reranker)
- PostgreSQL + pgvector + pgvectorscale as a database-native index
- IPFS ModelPack distribution for model artifacts (manifest CID = model version)
- Receipts + signatures for verified memory tiers (CID + wallet signature + quorum)

---

## Credits / acknowledgements
- Stanford Alpaca work influenced common instruction tuning patterns; this repo uses a lightweight Alpaca-style training scaffold where applicable.

---

## License
See `LICENSE` (if present) and repository headers.
