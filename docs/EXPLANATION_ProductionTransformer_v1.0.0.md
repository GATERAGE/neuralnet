# EXPLANATION · `production_transformer.py` (Production v1.0.0)

**Repository:** `GATERAGE/neuralnet`  
**File:** `production_transformer.py`  
**Version:** **v1.0.0**  
**Last updated:** 2026-02-14

This document explains the **ProductionTransformer v1.0.0** implementation: a compact, decoder-style Transformer intended for **local prototyping** and **RAGE wiring**, with practical “production-adjacent” upgrades (pre-norm, safer masking, optional KV cache, optional SDPA).

> **Scope:** logits-only language model core (decoder-style).  
> **Non-goals:** tokenizer, full decoding strategies, distributed training, SOTA optimizations beyond core correctness/perf helpers.

---

## What changed from the earlier minimal version

v1.0.0 adds:
- **Pre-norm blocks** (LayerNorm before attention/FFN) for improved stability.
- **Masking semantics clarified**:
  - `attn_mask` is a **boolean allow mask** (True=allowed, False=masked), matching PyTorch SDPA docs. 
- **Padding support** via `key_padding_mask` (True=pad).
- **Causal + padding mask composition** (logical AND).
- **Optional KV-cache** for fast autoregressive decoding.
- **Optional SDPA** (`torch.nn.functional.scaled_dot_product_attention`) with safe dropout handling.

---

## Mask semantics (critical)

This implementation uses **allow masks**:

- `allow_mask == True` → participates in attention
- `allow_mask == False` → masked (treated as -∞ bias before softmax)

This matches the SDPA documentation pseudocode and parameter description. citeturn1view0

`key_padding_mask` follows the common convention:
- `True` = padding (masked)
- `False` = real token (allowed)

---

## Components overview

### `ProductionTransformerConfig`
A dataclass capturing model hyperparameters and feature flags:
- `use_sdpa`: use SDPA when available
- `final_layer_norm`: optional final LN
- `activation`: `gelu` or `relu`

### `PositionalEncoding`
Non-trainable sinusoidal positional encodings.

### `MultiHeadSelfAttention`
- supports SDPA when available
- supports manual attention fallback
- supports KV-cache: concatenates past K/V with new K/V

### `TransformerBlock` (Pre-norm)
- `x = x + dropout(Attn(LN(x)))`
- `x = x + dropout(FFN(LN(x)))`

### `ProductionTransformer`
- builds attention masks once per forward
- runs stacked blocks
- returns logits, and optionally `present_key_values` for caching.

---

## API

### Forward
```python
logits = model(input_ids)

# or with cache
logits, past = model(input_ids, use_cache=True)
logits, past = model(next_token_ids, causal=False, use_cache=True, past_key_values=past)
```

### Greedy generate (KV-cache)
```python
out_ids = model.greedy_generate(prompt_ids, max_new_tokens=64, eos_token_id=None)
```

---

## Known limitations (still true in v1.0.0)

- No tokenizer / text decoding.
- No sampling (top-k, top-p, temperature), only greedy helper.
- Cache masking assumes typical generation (no padding during decode).
- Sinusoidal PE (not RoPE); good for a minimal core, not SOTA.
- Not optimized for very long context; consider RoPE + flash-attn oriented architectures for scaling.

---

## Recommended next upgrades (v1.1+)

- Add padding-aware cache handling if you need padded batched decoding.
- Add a proper generation module (sampling, repetition penalty, eos handling per batch).
- Add RoPE and RMSNorm (LLaMA-like) for modern performance.
- Add KV-cache packing + `scaled_dot_product_attention(is_causal=True)` fast path where applicable. citeturn1view0
