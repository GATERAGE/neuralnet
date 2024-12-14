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


# Retrieval-Augmented Generation with Configurable Chunk Size

This project provides a multi-format data ingestion pipeline (PDF, DOCX, TXT, MD, URLs), chunking them at a user-configurable size (e.g. 128 words, 4096 words, etc.), storing embeddings in FAISS, then augmenting user queries with retrieved chunks. The final prompt is routed to a local minimal Transformer or external LLM API (OpenAI, Together.ai, Ollama).

## Setup

1. **Install Python Dependencies**:
   ```bash
   pip install torch torchvision torchaudio
   pip install faiss-cpu             # or faiss-gpu
   pip install sentence-transformers requests
   pip install PyPDF2 python-docx    # for PDF / DOCX
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

   
   

   

