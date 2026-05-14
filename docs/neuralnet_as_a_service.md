# neuralnet as a Service

> # ⚠️ PROTOTYPE
>
> *This service is at **version 0.1.0a2** — an explicit PEP-440 alpha. The
> primitives below are documented as a target; the implementation is the
> reference scripts at the repo root and is **not yet stable**. Interfaces
> will change. Pin the exact commit if you build against it. The other
> three GATERAGE repos (RAGE, aglm, mastermind) are at 0.1.0+; this one
> is intentionally behind until the API stabilizes.*
>
> *0.1.0a2 cleanup: meta-files `optimized_transformer.py` + `ipfs_fetch.py`
> extracted into real importable modules `production_transformer_rage.py`
> + `ipfs_fetch_cli.py`. Dotted-name import in `generate.py` fixed.
> Dotted-name filename `production_transformer_v1.0.0.py` renamed to
> `production_transformer_v1.py`.*

Companion specs:

- [RAGE — Retrieval Augmented Generative Engine](https://github.com/GATERAGE/RAGE)
- [aGLM — Autonomous General Learning Model](https://github.com/GATERAGE/aglm)
- [MASTERMIND — Strategic Orchestrator](https://github.com/GATERAGE/mastermind)

Together the four form: **RAGE remembers, aGLM decides, MASTERMIND orchestrates, neuralnet trains and serves.**

---

## 1. What neuralnet is

`neuralnet` is the *training & serving* corner of the GATERAGE architecture.
Where RAGE provides retrieval and aGLM provides decision-making, neuralnet
provides the **actual model** that does the generation — and the
**content-addressed distribution mechanism** (ModelPack on IPFS) that
ships weights between peers without trusting a central host.

This repo is an **exploration** rather than a polished library. It serves
three audiences:

1. **Teaching audience** — three transformer implementations (minimal →
   stable v1.0.0 → RAGE-optimized v1.1.0) document the architecture
   trajectory.
2. **Reference implementation** — anyone wanting to build their own
   RAGE pipeline can read the scripts top to bottom and reproduce the
   pattern.
3. **Future production substrate** — once the API stabilizes, the
   scripts will be reorganized into a `neuralnet/` package that the
   other GATERAGE repos can `pip install neuralnet` against.

mindX is one consumer of the *pattern*; this repo is the canonical
agnostic home, with the caveat that the API is still moving.

---

## 2. The six modules (current state)

| Script | What it provides | Stability |
|---|---|---|
| `production_transformer.py` | Minimal decoder transformer (127 LOC, teaching version). | reference |
| `production_transformer_v1.py` *(was `v1.0.0.py`)* | Cleaned-up: pre-norm, bool allow-masks, optional SDPA, optional KV cache. | reference |
| `production_transformer_rage.py` *(new in 0.1.0a2, extracted from `optimized_transformer.py`)* | RAGE-flavored v1.1.0: RMSNorm + SwiGLU + GQA + RoPE + ModelPack loader. **Canonical importable.** | prototype |
| `optimized_transformer.py` | DEPRECATED meta-file (kept for history). Use `production_transformer_rage.py` for imports. | deprecated |
| `llm_router.py` | `LLMRouter` — pluggable backend router across `local` / `openai` / `together` / `ollama`. | prototype |
| `rag_inference.py` | `RAGInference` — composes `RAGEDataLoader` + FAISS + `LLMRouter` into one orchestrator. | prototype |
| `rage_dataloader.py` | `RAGEDataLoader` — chunks `.txt`/`.md`/`.pdf`/`.docx` and remote URLs. | prototype |
| `ipfs_fetch_cli.py` *(new in 0.1.0a2, extracted from `ipfs_fetch.py`)* | ModelPack manifest + shard fetch by CID with sha256 verification + cache. **Canonical importable.** | prototype |
| `ipfs_fetch.py` | DEPRECATED meta-file (kept for history). Use `ipfs_fetch_cli.py` for imports. | deprecated |
| `simplemind_torch.py` | Small MLP reranker (PyTorch). "Policy brain" that decides which retrieved chunks to include. | prototype |
| `simplemind_jax.py` | Same in JAX (offline training option). | prototype |
| `server.js` | Node.js HTTP server. Endpoints: `POST /ingest`, `POST /inference`. | prototype |
| `train.py` / `generate.py` | Training loop + generation entrypoint for the local transformer. | exploratory |

Read [`TECHNICAL.md`](TECHNICAL.md) for the architecture diagram and the
full data-flow walk-through. Read [`SIMPLEMIND_IN_RAGE.md`](SIMPLEMIND_IN_RAGE.md)
for how the reranker fits between FAISS retrieval and prompt assembly.
Read [`PROMOTE_MODELPACK.md`](PROMOTE_MODELPACK.md) for the IPFS
ModelPack publishing workflow.

---

## 3. The relationship to the GATERAGE triangle

```
                ┌──────────────────────┐
                │     MASTERMIND       │   directive → plan → execute
                └─────────┬────────────┘
                          │ delegates to
                          ▼
            ┌────────────────────────────┐
            │           aGLM             │   Perceive-Orient-Decide-Act
            │                            │   + BeliefSystem
            └─────────┬──────────────────┘
                      │ retrieves via       │ calls model via
                      ▼                      ▼
            ┌──────────────────┐  ┌─────────────────────────┐
            │       RAGE       │  │       neuralnet         │  ← this repo
            │  (retrieval)     │  │  ProductionTransformer  │
            │                  │  │  LLMRouter (4 backends) │
            │                  │  │  SimpleMind reranker    │
            │                  │  │  IPFS ModelPack         │
            └──────────────────┘  └─────────────────────────┘
```

- **RAGE** is `pip install rage` — stable substrate for retrieval.
- **aGLM** is `pip install aglm` — stable decision primitives.
- **MASTERMIND** is `pip install mastermind` — stable orchestration.
- **neuralnet** is `pip install -e git+https://github.com/GATERAGE/neuralnet@<commit>` — **PROTOTYPE** model substrate (pin the commit; the API moves).

---

## 4. The HTTP surface (server.js)

When `node server.js` is running:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | minimal HTML UI |
| `POST` | `/ingest` | spawn Python: chunk + embed + add to FAISS index |
| `POST` | `/inference` | spawn Python: retrieve top-k + augment prompt + call LLMRouter |

Outputs land in `memory/` as audit logs — every ingest and every
inference is recoverable.

---

## 5. The programmatic API (current shape)

```python
# After `pip install -e ".[torch,rage,data,serve,ipfs]"`

from rag_inference import RAGInference

rag = RAGInference(
    data_folder="docs",
    index_path="faiss_index",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=128,
    llm_backend="local",  # or "openai" / "together" / "ollama"
)
rag.build_or_load_index()
answer = rag.run_inference("What is the RAGE pattern?")
```

```python
from llm_router import LLMRouter

router = LLMRouter()
text = router.generate("Explain RAGE in one sentence.", backend="ollama")
```

```python
from production_transformer import ProductionTransformer
import torch

model = ProductionTransformer(vocab_size=10000)
model.load_state_dict(torch.load("transformer_checkpoint.pt"))
model.eval()
```

```bash
# ModelPack distribution: fetch a content-addressed model by CID.
python ipfs_fetch.py fetch --manifest-cid <CID> --out-dir ./models
```

---

## 6. ModelPack — universal access via IPFS

The most original piece of `neuralnet` is `ipfs_fetch.py`. Its claim:

> *Any node can fetch the exact model by CID. Integrity: CIDs prove
> content immutability; hashes verify downloads. Scalability: shard-level
> caching allows global distribution.*

The convention:

1. **Export weights** as `safetensors`, sharded as
   `model-00001-of-000NN.safetensors`.
2. **Compute SHA-256** for each shard.
3. **Add to IPFS** to get CIDs.
4. **Publish a manifest** listing `[{filename, cid, sha256}]`.
5. **Consumers fetch** by manifest CID; each shard's sha256 is verified
   on download; weights load only when verified.

See [`PROMOTE_MODELPACK.md`](PROMOTE_MODELPACK.md) for the full
workflow.

This is the piece that lets RAGE/aGLM/MASTERMIND-based systems distribute
trained models peer-to-peer without trusting a registry. The cypherpunk2048
*no-trapdoors rule* applied to model weights.

---

## 7. Service boundaries (target — not yet enforced)

When stable, neuralnet should **not**:

- Hold private keys.
- Custodize models. ModelPack publishing is opt-in; weights are always
  content-addressed.
- Bypass the GATERAGE triangle. neuralnet is the model layer.
- Pin to one transformer. The three transformer files are intentional
  versions; consumers swap freely.

When stable, neuralnet should:

- Provide a typed ProductionTransformer class with stable signatures.
- Provide the four-backend LLMRouter.
- Provide the FAISS-backed RAGInference orchestrator.
- Provide SimpleMind reranking for retrieval policy.
- Provide IPFS ModelPack publishing + fetching.
- Stay framework-agnostic on the LLM router.

---

## 8. Roadmap to 1.0

| Phase | What lands | When |
|---|---|---|
| **neuralnet-0.1.0a1** | Apache-2.0 + pyproject.toml + this spec + tests + typo fixes | Shipped 2026-05-14 |
| **neuralnet-0.1.0a2** | Meta-file extraction + dotted-name fixes + deprecation headers + expanded tests | Shipped 2026-05-14 |
| **neuralnet-0.1.0a3** | Reorganize flat scripts into `neuralnet/` Python package | next |
| **neuralnet-0.2.0** | Stable `LLMRouter` API + integration tests against real Ollama | next |
| **neuralnet-0.3.0** | `ProductionTransformer` consolidates to one canonical version | next |
| **neuralnet-0.4.0** | ModelPack publishing CLI (`neuralnet pack publish`) + verifier | next |
| **neuralnet-0.5.0** | SimpleMind training pipeline + integration test against RAGE.retrieval | next |
| **neuralnet-1.0.0** | API frozen; stable consumer surface; production-ready | when ready |

Until 1.0, treat every commit as breaking. Pin by SHA.

---

## 9. Known issues / fixes

### 0.1.0a2 (this release)

- ✅ **Fixed**: meta-file `optimized_transformer.py` extracted into the real
  module `production_transformer_rage.py` (501 LOC; classes: `RAGETransformerConfig`,
  `ModelPack`, `RMSNorm`, `RotaryEmbedding`, `SwiGLU`, `GQASelfAttention`,
  `DecoderBlock`, `ProductionTransformerRAGE`, helpers).
- ✅ **Fixed**: meta-file `ipfs_fetch.py` extracted into the real module
  `ipfs_fetch_cli.py` (250 LOC). Both originals kept with a prepended
  `# DEPRECATED — META-FILE` header for git history.
- ✅ **Fixed**: `generate.py` no longer imports `production_transformer_rage_v1.1.0`
  (an illegal dotted module name) — now imports from `production_transformer_rage`.
- ✅ **Fixed**: `production_transformer_v1.0.0.py` renamed to
  `production_transformer_v1.py` so it imports cleanly.

### 0.1.0a1

- ✅ **Fixed**: `llm_router.py` defaulted Ollama port to `11411` (typo); upstream is `11434`.
- ✅ **Fixed**: `install.rage` had `python3.1\`` typo in the venv-creation line.

### Open (planned for 0.2.0+)

- ⚠️ Reorganize flat scripts into a `neuralnet/` Python package.
- ⚠️ No integration tests for heavy paths (FAISS, sentence-transformers, transformer forward) — smoke tests only. Adding in 0.2.0.
- ⚠️ `server.js` spawns `python` with assumptions about cwd; move to subprocess-relative paths.
- ⚠️ The two `optimized_transformer.py` / `ipfs_fetch.py` meta-files will be removed entirely in 0.2.0 once the deprecation window closes.
- ⚠️ Normalize LICENSE to pure Apache-2.0 (currently a small custom file).

---

## 10. References

- [`README.md`](../README.md) — repo overview
- [`TECHNICAL.md`](TECHNICAL.md) — full architecture
- [`SIMPLEMIND_IN_RAGE.md`](SIMPLEMIND_IN_RAGE.md) — policy/reranking brain
- [`PROMOTE_MODELPACK.md`](PROMOTE_MODELPACK.md) — IPFS ModelPack publishing
- [`EXPLANATION.md`](EXPLANATION.md) — design rationale
- [`PRODUCTION_TRANSFORMER.md`](../PRODUCTION_TRANSFORMER.md) — class-level transformer doc
- [RAGE](https://github.com/GATERAGE/RAGE), [aglm](https://github.com/GATERAGE/aglm), [mastermind](https://github.com/GATERAGE/mastermind)
- [mindX](https://github.com/agenticplace) — production consumer of the pattern
- [rage.pythai.net](https://rage.pythai.net) — RAGE/aGLM/MASTERMIND/neuralnet docs
