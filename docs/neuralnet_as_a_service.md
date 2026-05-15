# neuralnet as a Service

> # ⚠️ PROTOTYPE
>
> *This service is at **version 0.1.0a7** — an explicit PEP-440 alpha.
> Interfaces will still change. Pin the exact commit if you build against
> this. The other three GATERAGE repos (RAGE, aglm, mastermind) are at
> 0.1.0+; this one is intentionally behind until 1.0.*
>
> *0.1.0a7 closes the Node.js test gap. `tests/test_server_js.py` adds 7
> smoke tests: static-file route assertions, the `python -m neuralnet.inference`
> spawn contract (no references to the removed `rag_inference.py` shim,
> `cwd: __dirname` pinned), a `node --check` syntax pass, and two live-start
> tests that actually launch `node server.js` on an OS-assigned free port
> and verify GET / serves HTML + that unknown routes 404. To make the live
> tests bindable on shared CI hosts, `server.js`'s `PORT` is now
> env-overridable: `NEURALNET_PORT` wins over `PORT` wins over `3000`.
> Full suite: 31/31 passing.*
>
> *Canonical imports (unchanged from 0.1.0a3):*
>
> ```python
> from neuralnet import ProductionTransformer, ProductionTransformerRAGE
> from neuralnet.router import LLMRouter
> from neuralnet.inference import RAGInference
> from neuralnet.modelpack import fetch_modelpack
> ```

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

## 2. The package (since v0.1.0a3)

The canonical code lives in the `neuralnet/` Python package. The
top-level scripts that shipped through v0.1.0a2 are now deprecation
shims that re-export from the package and emit a `DeprecationWarning`
at import time.

### 2.1 Canonical package modules

| Module | What it provides | Stability |
|---|---|---|
| `neuralnet.transformer` | Minimal decoder transformer (127 LOC, teaching version). Classes: `PositionalEncoding`, `MultiHeadSelfAttention`, `TransformerBlock`, `ProductionTransformer`. | reference |
| `neuralnet.transformer_v1` | Cleaned-up single-file version: pre-norm, bool allow-masks, optional SDPA, optional KV cache. | reference |
| `neuralnet.transformer_rage` | RAGE-flavored v1.1.0: RMSNorm + SwiGLU + GQA + RoPE + ModelPack loader. Classes: `RAGETransformerConfig`, `ModelPack`, `ProductionTransformerRAGE`. | prototype |
| `neuralnet.router` | `LLMRouter` — pluggable backend router across `local` / `openai` / `together` / `ollama`. | prototype |
| `neuralnet.inference` | `RAGInference` — composes `RAGEDataLoader` + FAISS + `LLMRouter` into one orchestrator. | prototype |
| `neuralnet.dataloader` | `RAGEDataLoader` — chunks `.txt`/`.md`/`.pdf`/`.docx` and remote URLs. | prototype |
| `neuralnet.simplemind` | Small MLP reranker (PyTorch). "Policy brain" that decides which retrieved chunks to include. | prototype |
| `neuralnet.modelpack` | ModelPack manifest + shard fetch by CID with sha256 verification + cache. | prototype |

The `neuralnet/__init__.py` lazily re-exports the heavy classes
(`ProductionTransformer`, `ProductionTransformerRAGE`, `RAGETransformerConfig`,
`ModelPack`, `LLMRouter`, `RAGInference`, `RAGEDataLoader`) so
`from neuralnet import ProductionTransformer` works once `torch` is
installed. Each `try/except ImportError` block keeps `import neuralnet`
from failing when heavy deps are absent — the relevant class just
doesn't appear in `__all__`.

### 2.2 Migration table (top-level shims removed in 0.1.0a5)

The 8 top-level deprecation shims that shipped in 0.1.0a3 were removed in
0.1.0a5 after two alpha cycles. The mapping consumers should follow:

| ❌ Removed top-level import | ✅ Use instead |
|---|---|
| `from production_transformer import ...` | `from neuralnet.transformer import ...` |
| `from production_transformer_v1 import ...` | `from neuralnet.transformer_v1 import ...` |
| `from production_transformer_rage import ...` | `from neuralnet.transformer_rage import ...` |
| `from llm_router import ...` | `from neuralnet.router import ...` |
| `from rag_inference import ...` | `from neuralnet.inference import ...` |
| `from rage_dataloader import ...` | `from neuralnet.dataloader import ...` |
| `from simplemind_torch import ...` | `from neuralnet.simplemind import ...` |
| `from ipfs_fetch_cli import ...` | `from neuralnet.modelpack import ...` |

In 0.1.0a3 these imports still worked with a `DeprecationWarning`. In
0.1.0a5 they raise `ImportError`. The migration is a sed.

### 2.3 Files at root that are NOT in the package

| File | Why |
|---|---|
| `simplemind_jax.py` | Alternate JAX implementation of the reranker. Kept flat because the package picks the PyTorch variant; the JAX one is for offline training. |
| `server.js`, `index.html`, `style.css` | Node.js UI server. Not Python; not part of the package. |
| `train.py`, `generate.py` | Top-level entrypoint scripts. Use the package internally. |
| `install.rage` | Bash installer. |

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
| **neuralnet-0.1.0a3** | Reorganize flat scripts into `neuralnet/` Python package | Shipped 2026-05-14 |
| **neuralnet-0.1.0a4** | Remove deprecated meta-files; 8 integration tests for heavy paths | Shipped 2026-05-14 |
| **neuralnet-0.1.0a5** | Remove top-level deprecation shims (close the alpha window) | Shipped 2026-05-14 |
| **neuralnet-0.1.0a6** | Apache-2.0 LICENSE normalization + `server.js` cwd-safety + README rewrite | Shipped 2026-05-14 |
| **neuralnet-0.1.0a7** | `tests/test_server_js.py` smoke test for the Node.js routes; env-overridable `PORT` (`NEURALNET_PORT`) | Shipped 2026-05-14 |
| **neuralnet-0.1.0a8** | Document `simplemind_jax.py` JAX-side parity (or fold the JAX path into `neuralnet.simplemind_jax`) | next |
| **neuralnet-0.2.0** | Stable `LLMRouter` API + integration tests against real Ollama | next |
| **neuralnet-0.3.0** | `ProductionTransformer` consolidates to one canonical version | next |
| **neuralnet-0.4.0** | ModelPack publishing CLI (`neuralnet pack publish`) + verifier | next |
| **neuralnet-0.5.0** | SimpleMind training pipeline + integration test against RAGE.retrieval | next |
| **neuralnet-1.0.0** | API frozen; stable consumer surface; production-ready | when ready |

Until 1.0, treat every commit as breaking. Pin by SHA.

---

## 9. Known issues / fixes

### 0.1.0a7 (this release)

- ✅ **Done**: `tests/test_server_js.py` — 7 smoke tests for the Node.js UI:
  file presence, expected route handlers (GET `/`, GET `/style.css`,
  POST `/ingest`, POST `/inference`), the post-0.1.0a5 spawn contract
  (`python -m neuralnet.inference` with `cwd: __dirname`, no
  `rag_inference.py`), the 0.1.0a7 `NEURALNET_PORT` env contract, a
  `node --check` syntax pass, and two live-start tests that `Popen(node
  server.js)` on an OS-assigned free port via `NEURALNET_PORT`, GET / +
  assert 200 + HTML, GET unknown + assert 404, SIGTERM cleanup in finally.
- ✅ **Done**: `server.js` `PORT` is now env-overridable —
  `NEURALNET_PORT` wins over `PORT` wins over `3000`. This is what makes
  the live-start tests possible and lets users running the UI alongside
  other :3000 services move it without editing the source.
- ✅ **Net effect**: full suite is now 31 tests (16 smoke + 8 integration
  + 7 server.js). The Node.js layer has its first smoke/syntax/live tests.

### 0.1.0a6

- ✅ **Done**: LICENSE normalized to full Apache-2.0 text. The previous file
  was a 1058b custom MIT notice, even though every SPDX header and the
  pyproject classifier said Apache-2.0 since 0.1.0a1. The mismatch is now
  resolved.
- ✅ **Done**: `server.js` was still spawning the removed `rag_inference.py`
  shim (broken since 0.1.0a5). Both `/ingest` and `/inference` routes now
  spawn `python -m neuralnet.inference` with `cwd: __dirname`, so the
  server works regardless of where it's launched from.
- ✅ **Done**: README rewritten end-to-end. The old README dated from 2024
  and had accumulated stacked banners. The new one leads with the 0.1.0a5
  layout, the four-corner GATERAGE architecture, install + quick-use
  examples, and a pointer to the spec.

### 0.1.0a5

- ✅ **Done**: removed the 8 top-level deprecation shims
  (`production_transformer.py`, `production_transformer_v1.py`,
  `production_transformer_rage.py`, `llm_router.py`, `rag_inference.py`,
  `rage_dataloader.py`, `simplemind_torch.py`, `ipfs_fetch_cli.py`).
  Two alpha cycles of `DeprecationWarning` is enough — consumers now
  import from `neuralnet.*` directly.
- ✅ **Done**: dropped `[tool.setuptools] py-modules` from `pyproject.toml`
  (no more loose top-level Python modules to install).
- ✅ **Done**: smoke tests updated: `test_top_level_shims_exist` →
  `test_top_level_shims_removed`; new `test_top_level_python_layout`
  pins the exact-3-file root state (`generate.py`, `train.py`,
  `simplemind_jax.py`).
- ✅ **Net effect**: repo went from 18 top-level Python files (0.1.0a1)
  → 3 top-level Python files (0.1.0a5). Everything else lives in
  `neuralnet/` or under `docs/` / `tests/`.

### 0.1.0a4

- ✅ **Done**: removed the two deprecated meta-files `optimized_transformer.py`
  and `ipfs_fetch.py`. Their content has lived at `neuralnet.transformer_rage`
  and `neuralnet.modelpack` since 0.1.0a2 — two alpha cycles is enough.
- ✅ **Done**: added `tests/test_integration.py` — 8 tests gated by
  `pytest.importorskip("torch")` etc. Coverage:
    - `ProductionTransformer` forward + causal-mask shape
    - `ProductionTransformerRAGE` forward (GQA + RoPE + RMSNorm + SwiGLU)
    - `SimpleMindTorch` reranker forward
    - `LLMRouter` construction (Ollama default port check)
    - `RAGEDataLoader._chunk_text` chunking (×2)
    - `neuralnet.modelpack init-template` CLI

### 0.1.0a3

- ✅ **Done**: flat scripts reorganized into a `neuralnet/` Python package
  (8 canonical modules: `transformer`, `transformer_v1`, `transformer_rage`,
  `router`, `inference`, `dataloader`, `simplemind`, `modelpack`).
- ✅ **Done**: top-level files retained as deprecation shims; removal in 0.1.0a5.
- ✅ **Done**: cross-module imports inside the package use relative form.

### 0.1.0a2

- ✅ **Fixed**: meta-file `optimized_transformer.py` extracted into the real
  module `production_transformer_rage.py` (501 LOC; now lives at
  `neuralnet/transformer_rage.py`).
- ✅ **Fixed**: meta-file `ipfs_fetch.py` extracted into the real module
  `ipfs_fetch_cli.py` (250 LOC; now lives at `neuralnet/modelpack.py`).
- ✅ **Fixed**: `generate.py` no longer imports `production_transformer_rage_v1.1.0`.
- ✅ **Fixed**: `production_transformer_v1.0.0.py` renamed.

### 0.1.0a1

- ✅ **Fixed**: `llm_router.py` defaulted Ollama port to `11411` (typo); upstream is `11434`.
- ✅ **Fixed**: `install.rage` had `python3.1\`` typo in the venv-creation line.

### Open (planned for 0.1.0a5 / 0.2.0)

- ⚠️ Remove the top-level deprecation shims after one alpha cycle (planned
  for 0.2.0).
- ⚠️ No integration tests for heavy paths (FAISS, sentence-transformers,
  transformer forward) — smoke tests only. Adding in 0.1.0a4.
- ⚠️ `server.js` spawns `python` with assumptions about cwd; move to
  subprocess-relative paths.
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
