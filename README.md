# <a href="https://rage.pythai.net/production-transformer-2026/">neuralnet</a>

> ## ⚠️ PROTOTYPE — version 0.1.0a6 (PEP-440 alpha)
>
> Interfaces will change. Pin the exact commit if you build against this.
> The companion GATERAGE repos ([RAGE](https://github.com/GATERAGE/RAGE),
> [aglm](https://github.com/GATERAGE/aglm), [mastermind](https://github.com/GATERAGE/mastermind))
> are at 0.1.0+; this one is intentionally behind until the API stabilizes.
>
> Service spec + roadmap to 1.0: [`docs/neuralnet_as_a_service.md`](docs/neuralnet_as_a_service.md).

**neuralnet is the *training & serving* corner of the [GATERAGE](https://github.com/GATERAGE) architecture.** The full slogan: *RAGE remembers, aGLM decides, MASTERMIND orchestrates, **neuralnet trains and serves**.*

## What it provides

The `neuralnet/` Python package ships three transformer implementations and the surrounding RAGE pipeline:

- **Three transformer variants** documenting the architecture trajectory:
  - `neuralnet.transformer` — minimal teaching version (127 LOC; PositionalEncoding + MultiHeadSelfAttention + TransformerBlock + ProductionTransformer)
  - `neuralnet.transformer_v1` — cleaned single-file (pre-norm, bool allow-masks, optional SDPA, optional KV cache)
  - `neuralnet.transformer_rage` — RAGE-flavored v1.1 (RMSNorm + SwiGLU + GQA + RoPE + ModelPack loader)
- **`neuralnet.router`** — `LLMRouter` across `local` / `openai` / `together` / `ollama`
- **`neuralnet.inference`** — `RAGInference` orchestrator (FAISS + retrieval + LLMRouter)
- **`neuralnet.dataloader`** — `RAGEDataLoader` (chunking txt / md / json / py / ts / html / pdf / docx + URLs)
- **`neuralnet.simplemind`** — small MLP reranker (PyTorch) — the policy brain that decides which retrieved chunks to include
- **`neuralnet.modelpack`** — IPFS ModelPack manifest + shard fetch with sha256 verification

Plus three top-level entrypoint scripts:
- `generate.py` — generation entrypoint
- `train.py` — training loop
- `simplemind_jax.py` — alternate JAX reranker (offline training)

And a minimal Node.js UI server:
- `server.js`, `index.html`, `style.css` — POST `/ingest` and POST `/inference`

## Install

```bash
pip install .                  # core only
pip install ".[torch]"         # PyTorch + numpy (transformers, reranker)
pip install ".[rage]"          # github.com/GATERAGE/RAGE + sentence-transformers + FAISS
pip install ".[data]"          # PyPDF2 + python-docx + requests
pip install ".[serve]"         # openai + requests (for LLMRouter cloud backends)
pip install ".[ipfs]"          # requests (for ModelPack fetch)
pip install ".[jax]"           # jax + optax (for the JAX reranker only)
pip install ".[all]"           # everything except [jax] and [dev]
pip install ".[dev]"           # pytest + ruff
```

For the Ollama local backend you also need [Ollama](https://ollama.com):

```bash
ollama serve
ollama run deepseek-r1:1.5b
```

## Quick use

### Forward a transformer

```python
import torch
from neuralnet import ProductionTransformer

model = ProductionTransformer(vocab_size=10000, d_model=128, num_heads=4, num_layers=2)
tokens = torch.randint(0, 10000, (1, 32))
logits = model(tokens)  # (1, 32, 10000)
```

### Run the full RAGE pipeline

```python
from neuralnet.inference import RAGInference

rag = RAGInference(
    data_folder="docs",
    index_path="faiss_index",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=128,
    llm_backend="ollama",  # or "local" / "openai" / "together"
)
rag.build_or_load_index()
answer = rag.run_inference("What is the RAGE pattern?")
```

### Run the Node.js UI server

```bash
bash install.rage              # create rage venv + pip install -e .
node server.js                 # http://localhost:3000
```

`server.js` spawns `python -m neuralnet.inference` for ingest + inference, with `cwd` pinned to its own directory (so it works regardless of where the server is launched from).

### Fetch a ModelPack from IPFS

```bash
python -m neuralnet.modelpack init-template --out MODEL_PACK.json
python -m neuralnet.modelpack fetch --manifest-cid <CID> --out-dir ./models
```

Each shard's sha256 is verified on download; weights load only when verified. The full ModelPack workflow is in [`docs/PROMOTE_MODELPACK.md`](docs/PROMOTE_MODELPACK.md).

## The four-corner GATERAGE architecture

```
                  ┌──────────────────────┐
                  │     MASTERMIND       │   directive → plan → execute
                  │  (orchestrator)      │   github.com/GATERAGE/mastermind
                  └─────────┬────────────┘
                            │ delegates to
                            ▼
            ┌────────────────────────────────────┐
            │              aGLM                  │   Perceive-Orient-Decide-Act
            │   (decision substrate)             │   + BeliefSystem
            │                                    │   github.com/GATERAGE/aglm
            └────────┬──────────────┬────────────┘
                     │ retrieves    │ calls model via
                     ▼              ▼
        ┌──────────────────┐   ┌─────────────────────────┐
        │       RAGE       │   │         neuralnet       │   (this repo)
        │ (retrieval       │   │  ProductionTransformer  │
        │  substrate)      │   │  ProductionTransformerRAGE
        │ github.com/      │   │  LLMRouter              │
        │ GATERAGE/RAGE    │   │  RAGInference           │
        └──────────────────┘   │  SimpleMind reranker    │
                               │  IPFS ModelPack         │
                               └─────────────────────────┘
```

The first three are stable. This repo is intentionally behind at PROTOTYPE 0.1.0a6. The companion article on rage.pythai.net documents the current state in detail: [production_transformer.py in 2026](https://rage.pythai.net/production-transformer-2026/).

## Tests

```bash
pip install ".[dev]"
pytest -v
```

24 tests:
- **16 smoke** (always run; no heavy deps required)
- **8 integration** gated by `pytest.importorskip("torch")` — run when torch + faiss + sentence-transformers + openai are present, skip cleanly when absent

## Background

`neuralnet` started as an exploration of RAGE training integration with a mini transformer. Thanks to the team at Stanford for the work on Alpaca and to the Professor-Codephreak / easyAGI lineage for the foundational architecture (MASTERMIND coordination, aGLM autonomous learning, RAGE retrieval).

The historical RAGE / aGLM / MASTERMIND philosophy papers live at:
- [`github.com/GATERAGE/RAGE/blob/main/ragepaper.md`](https://github.com/GATERAGE/RAGE/blob/main/ragepaper.md)
- [`github.com/GATERAGE/aglm`](https://github.com/GATERAGE/aglm) — root-level research files
- [`github.com/GATERAGE/RAGE/blob/main/mastermind.md`](https://github.com/GATERAGE/RAGE/blob/main/mastermind.md)

Project documentation is at [rage.pythai.net](https://rage.pythai.net).

## License

Apache-2.0. (c) 2024-2026 GATERAGE / Professor Codephreak.

Normalized from a small custom MIT file to full Apache-2.0 text in v0.1.0a6.

## Status & roadmap

See [`docs/neuralnet_as_a_service.md`](docs/neuralnet_as_a_service.md) for:
- Detailed module breakdown
- Service contract
- Migration table for any code still using pre-0.1.0a3 imports
- Roadmap from 0.1.0a6 → 1.0.0
