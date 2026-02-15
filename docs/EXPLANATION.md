# EXPLANATION · `production_transformer.py`
**Repository:** `GATERAGE/neuralnet`  
**File:** `production_transformer.py`  
**Last audited:** 2026-02-14

This document explains (and audits) the **minimal local Transformer** included in `neuralnet`. It is intentionally small and readable: good for learning, wiring prototypes, and replacing with a more capable model later.

> **Scope:** This module defines a decoder-style Transformer suitable for **language modeling logits** (next-token prediction) when given **token IDs**.  
> **Non-goals:** tokenizer, dataset pipeline, generation loop with KV-cache, mixed-precision training optimizations, distributed training, or SOTA architecture details.

---

## 1) What the model is

`ProductionTransformer` is a compact, GPT-like (decoder-only) Transformer made from:

- **Token embedding** (`nn.Embedding`)
- **Sinusoidal positional encoding** (`PositionalEncoding`)
- A stack of **Transformer blocks** (`TransformerBlock`)
  - multi-head self-attention (`MultiHeadSelfAttention`)
  - feed-forward network (2-layer MLP)
  - residual connections + layer normalization
- A **language-model head** (`lm_head`) projecting hidden states → vocabulary logits
- Optional **weight tying** (`lm_head.weight = embedding.weight`)

It returns **logits** with shape:

- **Input:** `x` = `(batch_size, seq_len)` (token indices)
- **Output:** `logits` = `(batch_size, seq_len, vocab_size)`

**Key concept:** the model is “decoder-like” because it supports **causal masking** to prevent attention to future tokens.

---

## 2) File inventory (classes & functions)

### 2.1 `PositionalEncoding`
Adds deterministic sine/cosine positional vectors to token embeddings.

- Precomputes `pe` with shape `(1, max_len, d_model)` and registers it as a buffer.
- On forward:
  - slices by current `seq_len`
  - adds positional encodings to `x`

**Forward shape**
- Input: `x` = `(B, T, D)`
- Output: `(B, T, D)`

**Important constraint**
- `T` must be `<= max_len`. If you pass longer sequences, you should increase `max_len` or implement dynamic extension.

---

### 2.2 `MultiHeadSelfAttention`
Implements **scaled dot-product attention** across `num_heads`.

**Projection**
- Q, K, V are computed by linear layers, each mapping `D → D`.
- Reshape + transpose creates head dimension:

| Tensor | Shape |
|---|---|
| Input `x` | `(B, T, D)` |
| `Q, K, V` after reshape | `(B, H, T, d_k)` where `d_k = D/H` |

**Attention scores**
- `scores = Q @ K^T / sqrt(d_k)`
- Shape: `(B, H, T, T)`

**Masking**
- If `mask` is provided, entries where `mask == 0` are set to `-inf` **before softmax**.

**Softmax + context**
- `attn_weights = softmax(scores, dim=-1)`
- `context = attn_weights @ V`
- context shape: `(B, H, T, d_k)` → reshaped back to `(B, T, D)` then projected by `self.out`.

---

### 2.3 `TransformerBlock`
A single block composed of:
1) Self-attention + residual + layer norm
2) Feed-forward network + residual + layer norm

**Current norm strategy:** **post-norm** (norm applied *after* residual).  
Modern large decoders often prefer **pre-norm** for stability, but post-norm is fine for small models and clarity.

**FFN**
- `Linear(D → dim_feedforward) → ReLU → Linear(dim_feedforward → D)`

---

### 2.4 `ProductionTransformer`
Composes:
- embedding + positional encoding
- `num_layers` blocks
- LM head

**Embedding scaling**
- The embedding output is multiplied by `sqrt(d_model)`; this is a common stabilization trick.

**Weight tying**
- If `tie_weights=True`, the embedding matrix and LM head weight share parameters (saves parameters and can improve performance).

---

### 2.5 `create_causal_mask(seq_len, device='cpu')`
Creates a **lower-triangular** mask of ones:

- Shape: `(1, seq_len, seq_len)`
- `1` means “allowed”, `0` means “masked/future”

This mask is broadcastable to attention scores `(B, H, T, T)`.

---

## 3) Tensor shapes: end-to-end walkthrough

Let:
- `B` = batch size
- `T` = sequence length
- `D` = `d_model`
- `H` = num heads
- `d_k = D / H`

### Step A — Embedding
- `x_ids`: `(B, T)`
- `x = embedding(x_ids)`: `(B, T, D)`
- `x *= sqrt(D)`

### Step B — Positional encoding
- `x = x + pe[:, :T, :]`

### Step C — For each layer
Attention:
- `Q,K,V`: `(B, H, T, d_k)`
- `scores`: `(B, H, T, T)`
- apply mask (optional)
- `attn_weights`: `(B, H, T, T)`
- `context`: `(B, H, T, d_k)` → `(B, T, D)`

FFN:
- `(B, T, D)` → `(B, T, dim_ff)` → `(B, T, D)`

### Step D — LM head
- `logits = lm_head(x)` → `(B, T, vocab_size)`

---

## 4) Audit: correctness and edge cases

### ✅ Works as intended for small-scale LM logits
- Attention math and head shaping are correct.
- Causal masking is compatible with the attention score shape.
- Weight tying is implemented in the standard way.

### ⚠️ Important caveats / improvements (recommended)

#### 4.1 Mask dtype and broadcasting
- The mask is currently a float tensor of 0/1 values; `mask == 0` is valid.
- **Recommendation:** use a boolean mask for clarity and to match modern PyTorch APIs.

Example:
```python
mask = torch.tril(torch.ones(T, T, device=device)).bool().unsqueeze(0)
```

#### 4.2 Mixed precision safety
Using `float('-inf')` can produce `nan` in some mixed precision contexts.
- **Recommendation:** use `torch.finfo(scores.dtype).min` instead of `-inf`:

```python
scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
```

#### 4.3 No padding mask helper
Only causal masking is provided. If you have padded sequences, you also need a **padding mask** (to prevent attention to pad tokens).
- **Recommendation:** support a combined `(causal AND not-pad)` mask.

#### 4.4 No KV-cache (generation efficiency)
For text generation, recomputing attention for all previous tokens each step is slow.
- **Recommendation:** add key/value caching for autoregressive decoding.

#### 4.5 Post-norm stability
Post-norm can become unstable as depth grows.
- **Recommendation for scaling:** use **pre-norm** blocks (LayerNorm before attention/FFN).

#### 4.6 Positional encoding max length
If `seq_len > max_len` you will slice beyond the buffer.
- **Recommendation:** increase `max_len` or implement a dynamic buffer expansion.

#### 4.7 Parameter initialization
Xavier init across all multi-dim parameters is fine for a minimal model.
- **Optional improvement:** follow common decoder init patterns (e.g., normal init for embeddings).

---

## 5) How to use it (practical examples)

### 5.1 Forward pass (logits)
```python
import torch
from production_transformer import ProductionTransformer, create_causal_mask

B, T = 2, 16
vocab_size = 10000

model = ProductionTransformer(vocab_size=vocab_size, d_model=128, num_heads=4, num_layers=2)
x = torch.randint(0, vocab_size, (B, T))

mask = create_causal_mask(seq_len=T, device=x.device)
logits = model(x, mask=mask)
print(logits.shape)  # (2, 16, 10000)
```

### 5.2 Training step (next-token prediction)
Language modeling uses **teacher forcing**:
- model sees tokens `x[:, :-1]`
- predicts tokens `x[:, 1:]`

```python
import torch
import torch.nn.functional as F
from production_transformer import ProductionTransformer, create_causal_mask

vocab_size = 10000
model = ProductionTransformer(vocab_size=vocab_size, d_model=128, num_heads=4, num_layers=2)
optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

B, T = 4, 64
x = torch.randint(0, vocab_size, (B, T))

inp = x[:, :-1]         # (B, T-1)
tgt = x[:, 1:]          # (B, T-1)
mask = create_causal_mask(inp.size(1), device=inp.device)

logits = model(inp, mask=mask)                # (B, T-1, V)
loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))

optim.zero_grad()
loss.backward()
optim.step()
print("loss:", float(loss))
```

### 5.3 Minimal greedy generation (no KV-cache)
This is slow but demonstrates correctness:

```python
import torch
from production_transformer import ProductionTransformer, create_causal_mask

def greedy_generate(model, prompt_ids, max_new_tokens=32):
    model.eval()
    out = prompt_ids.clone()

    for _ in range(max_new_tokens):
        T = out.size(1)
        mask = create_causal_mask(T, device=out.device)
        logits = model(out, mask=mask)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)

    return out

vocab_size = 10000
model = ProductionTransformer(vocab_size=vocab_size, d_model=128, num_heads=4, num_layers=2)

prompt = torch.randint(0, vocab_size, (1, 8))
gen = greedy_generate(model, prompt, max_new_tokens=16)
print(gen.shape)  # (1, 24)
```

---

## 6) Recommended modernization (if you want “production”)

If you intend this file to be more than an educational scaffold, here is a prioritized list:

### Priority A — correctness + stability upgrades
1) Switch mask to **bool** and use dtype-safe min values.
2) Add optional **padding mask** support.
3) Convert blocks to **pre-norm**.
4) Add final layer norm (common in decoders).

### Priority B — performance upgrades
1) Use PyTorch’s **`scaled_dot_product_attention`** (enables Flash Attention paths where available).
2) Add **KV-cache** for decoding.
3) Support `torch.compile()` for speed (PyTorch 2.x).

### Priority C — architecture upgrades (optional)
- Replace sinusoidal PE with **RoPE** (rotary position embeddings)
- Replace ReLU with **GELU** (common in modern LMs)
- Consider **RMSNorm** (common in LLaMA-style models)

---

## 7) Relationship to the rest of `neuralnet`

In this repo, `production_transformer.py` acts as a **local scaffold** for “generation.”  
The actual RAG pipeline (`rag_inference.py`) focuses on:
- chunking + embedding + FAISS retrieval
- building a prompt that includes “Relevant context”
- routing generation to selectable backends (`llm_router.py`)

So: this transformer is best viewed as a **local demo model**, not the core retrieval engine. citeturn2view2

---

## 8) Documentation audit notes

The existing `PRODUCTION_TRANSFORMER.md` is a good start, but for professional documentation you should:
- use ```python code fences (not ```bash) for Python examples
- explicitly state limitations: no tokenizer, no KV-cache, no padding mask helper
- clarify that it outputs **logits**, not decoded text
- include a short training example (next-token prediction)

(Those are all addressed in this document.) citeturn2view1turn3view0
