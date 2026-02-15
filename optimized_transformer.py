from pathlib import Path
from datetime import date

# 1) Create RAGE-optimized transformer code file
code = r'''#!/usr/bin/env python3
# production_transformer_rage.py
"""
ProductionTransformer-RAGE v1.1.0
A RAGE-optimized, decoder-style Transformer core designed for:
- correctness (mask semantics aligned with PyTorch SDPA)
- stability (pre-norm + RMSNorm)
- modern MLP (SwiGLU)
- scalable decoding (KV cache)
- portable distribution (optional ModelPack manifest + safetensors)

IMPORTANT DESIGN NOTES
- This is a lightweight core for universal access (CPU/VPS/edge).
- For high-throughput serving at scale, pair RAGE with a battle-tested runtime
  (e.g., vLLM / llama.cpp / TensorRT-LLM) and use the same IPFS ModelPack system.

Mask semantics
- `attn_mask` is a boolean "allow mask": True = allow attention, False = mask out.
  This matches torch.nn.functional.scaled_dot_product_attention semantics for bool masks.

Padding semantics
- `key_padding_mask` is boolean: True = PAD (mask), False = real token.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = "1.1.0"

Tensor = torch.Tensor
KVCache = Tuple[Tensor, Tensor]  # (k, v)
PastKeyValues = Optional[List[KVCache]]


# -----------------------------
# Config + lightweight ModelPack
# -----------------------------

@dataclass(frozen=True)
class RAGETransformerConfig:
    vocab_size: int = 32000
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 2             # grouped-query attention (GQA). set=1 for MQA
    dim_ff: int = 2048
    dropout: float = 0.0
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    tie_weights: bool = True
    use_sdpa: bool = True             # uses PyTorch SDPA when available
    final_norm: bool = True
    rmsnorm_eps: float = 1e-5


@dataclass(frozen=True)
class ModelPack:
    """
    IPFS-friendly distribution manifest (content-addressed).
    Store this JSON on IPFS; the manifest CID becomes the model's version pointer.

    Minimal manifest keys:
      - format: "safetensors"
      - shards: list of { "cid": "...", "filename": "...", "sha256": "..." }
      - config: model hyperparameters dict
      - tokenizer: optional CID/pointer (outside this scope)
    """
    format: str
    config: Dict[str, Union[int, float, str, bool]]
    shards: List[Dict[str, str]]
    # optional metadata:
    model_name: str = "ProductionTransformer-RAGE"
    model_version: str = __version__


# -----------------------------
# Utility: masks
# -----------------------------

def make_causal_allow_mask(q_len: int, k_len: int, device: Union[str, torch.device]) -> Tensor:
    """
    Boolean allow mask: True allowed, False masked.
    Shape: (1, 1, q_len, k_len) broadcastable to (B, H, q_len, k_len)
    """
    mask = torch.ones((q_len, k_len), dtype=torch.bool, device=device)
    mask = torch.tril(mask, diagonal=0)
    return mask.unsqueeze(0).unsqueeze(0)


def key_padding_to_allow_mask(key_padding_mask: Tensor, q_len: int) -> Tensor:
    """
    key_padding_mask: (B, k_len) with True=PAD.
    returns allow mask: (B, 1, q_len, k_len) with True=allowed.
    """
    if key_padding_mask.dtype != torch.bool:
        key_padding_mask = key_padding_mask.bool()
    allow_keys = ~key_padding_mask  # True where real tokens
    return allow_keys[:, None, None, :].expand(-1, 1, q_len, -1)


def combine_allow_masks(*masks: Optional[Tensor]) -> Optional[Tensor]:
    out: Optional[Tensor] = None
    for m in masks:
        if m is None:
            continue
        out = m if out is None else (out & m)
    return out


# -----------------------------
# RMSNorm + RoPE
# -----------------------------

class RMSNorm(nn.Module):
    """
    RMSNorm (Zhang et al., 2019): normalize by root mean square only (no mean subtraction).
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., D)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    """
    RoPE (rotary position embeddings) precomputes cos/sin for max_seq_len.
    """
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires head_dim to be even.")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, head_dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)  # (1,1,T,D)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: Tensor, k: Tensor, seq_offset: int = 0) -> Tuple[Tensor, Tensor]:
        """
        q, k: (B, H, T, D)
        seq_offset: starting position (for KV-cache decoding)
        """
        t = q.size(2)
        if seq_offset + t > self.max_seq_len:
            raise ValueError(f"RoPE positions exceed max_seq_len={self.max_seq_len}. Increase max_seq_len.")
        cos = self.cos_cached[:, :, seq_offset:seq_offset + t, :]
        sin = self.sin_cached[:, :, seq_offset:seq_offset + t, :]
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return (x * cos) + (rotate_half(x) * sin)


# -----------------------------
# Attention (GQA/MQA) + SwiGLU
# -----------------------------

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, dim_ff: int):
        super().__init__()
        # gate and up projections combined for speed
        self.w1 = nn.Linear(d_model, dim_ff * 2, bias=False)
        self.w2 = nn.Linear(dim_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.w1(x)
        x_gate, x_up = x.chunk(2, dim=-1)
        return self.w2(F.silu(x_gate) * x_up)


class GQASelfAttention(nn.Module):
    """
    Grouped-Query Attention:
      - Q has num_heads
      - K/V have num_kv_heads (<= num_heads)
    """
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, dropout: float, use_sdpa: bool, rope: RotaryEmbedding):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for GQA.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_repeat = num_heads // num_kv_heads
        self.dropout = dropout
        self.use_sdpa = use_sdpa and hasattr(F, "scaled_dot_product_attention")
        self.rope = rope

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

    def _shape_q(self, x: Tensor, bsz: int) -> Tensor:
        return x.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,T,D)

    def _shape_kv(self, x: Tensor, bsz: int) -> Tensor:
        return x.view(bsz, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B,Hkv,T,D)

    def _repeat_kv(self, x: Tensor) -> Tensor:
        # (B, Hkv, T, D) -> (B, H, T, D)
        return x.repeat_interleave(self.kv_repeat, dim=1)

    def forward(
        self,
        x: Tensor,
        allow_mask: Optional[Tensor],
        past_kv: Optional[KVCache],
        use_cache: bool,
        seq_offset: int,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        bsz, t, _ = x.size()

        q = self._shape_q(self.q_proj(x), bsz)      # (B,H,T,Dh)
        k = self._shape_kv(self.k_proj(x), bsz)     # (B,Hkv,T,Dh)
        v = self._shape_kv(self.v_proj(x), bsz)

        # Apply RoPE to Q and K (note: K is kv-heads; RoPE works per-head)
        q, k = self.rope(q, k, seq_offset=seq_offset)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)  # concat along sequence
            v = torch.cat([pv, v], dim=2)

        present_kv = (k, v) if use_cache else None

        # Repeat kv heads to match q heads for attention computation
        k_rep = self._repeat_kv(k)
        v_rep = self._repeat_kv(v)

        # allow_mask: (B,1,T,S) -> (B,H,T,S)
        if allow_mask is not None and allow_mask.size(1) == 1:
            allow_mask = allow_mask.expand(bsz, self.num_heads, allow_mask.size(2), allow_mask.size(3))

        if self.use_sdpa:
            dropout_p = self.dropout if self.training else 0.0
            # bool mask semantics: True means KEEP/attend
            attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=allow_mask, dropout_p=dropout_p, is_causal=False)
        else:
            scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,S)
            if allow_mask is not None:
                scores = scores.masked_fill(~allow_mask, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            attn_out = torch.matmul(attn, v_rep)  # (B,H,T,Dh)

        out = attn_out.transpose(1, 2).contiguous().view(bsz, t, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        return out, present_kv


class DecoderBlock(nn.Module):
    """
    Pre-norm block:
      x = x + Attn(RMSNorm(x))
      x = x + MLP(RMSNorm(x))
    """
    def __init__(self, cfg: RAGETransformerConfig, rope: RotaryEmbedding):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.attn = GQASelfAttention(cfg.d_model, cfg.num_heads, cfg.num_kv_heads, cfg.dropout, cfg.use_sdpa, rope)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.rmsnorm_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.dim_ff)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        allow_mask: Optional[Tensor],
        past_kv: Optional[KVCache],
        use_cache: bool,
        seq_offset: int,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        a, present = self.attn(self.norm1(x), allow_mask=allow_mask, past_kv=past_kv, use_cache=use_cache, seq_offset=seq_offset)
        x = x + self.dropout(a)
        m = self.mlp(self.norm2(x))
        x = x + self.dropout(m)
        return x, present


# -----------------------------
# Model
# -----------------------------

class ProductionTransformerRAGE(nn.Module):
    """
    RAGE-optimized decoder Transformer.

    Forward inputs:
      input_ids: (B, T)
      attention_mask: optional (B, T) keep mask (1/True=keep), used as padding info
      key_padding_mask: optional (B, T) True=pad
      causal: apply causal masking (recommended for training). For cached decoding with q_len=1, set causal=False.
      past_key_values: optional list of cached (k,v) per layer
      use_cache: return present_key_values

    Returns:
      logits OR (logits, present_key_values)
    """
    def __init__(self, cfg: Optional[RAGETransformerConfig] = None, **kwargs):
        super().__init__()
        self.cfg = cfg or RAGETransformerConfig(**kwargs)
        if self.cfg.d_model % self.cfg.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        head_dim = self.cfg.d_model // self.cfg.num_heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        self.head_dim = head_dim

        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=self.cfg.max_seq_len, theta=self.cfg.rope_theta)

        self.layers = nn.ModuleList([DecoderBlock(self.cfg, self.rope) for _ in range(self.cfg.num_layers)])
        self.final_norm = RMSNorm(self.cfg.d_model, eps=self.cfg.rmsnorm_eps) if self.cfg.final_norm else None
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)

        if self.cfg.tie_weights:
            self.lm_head.weight = self.embed.weight

        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "embed" not in name:
                nn.init.xavier_uniform_(p)

    def _build_allow_mask(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        causal: bool,
        past_key_values: PastKeyValues,
    ) -> Optional[Tensor]:
        device = input_ids.device
        bsz, t = input_ids.shape

        # Determine k_len (S) under cache
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_k, _ = past_key_values[0]  # (B, Hkv, S_past, Dh)
            s = past_k.size(2) + t
        else:
            s = t

        # Padding info:
        allow_from_padding = None
        if key_padding_mask is not None:
            if key_padding_mask.size() != (bsz, t):
                raise ValueError("key_padding_mask must be (B, T) for current tokens.")
            allow_from_padding = key_padding_to_allow_mask(key_padding_mask, q_len=t)

        # attention_mask (B,T) keep mask -> treat as padding
        allow_from_attention = None
        if attention_mask is not None:
            if attention_mask.dim() != 2 or attention_mask.size() != (bsz, t):
                raise ValueError("attention_mask must be (B, T) keep-mask (1/True=keep).")
            keep = attention_mask.bool()
            pad = ~keep
            allow_from_attention = key_padding_to_allow_mask(pad, q_len=t)

        # Causal (skip for cached decode where q_len=1 and s>1; no future keys exist)
        allow_from_causal = None
        if causal and not (past_key_values is not None and t == 1 and s > 1):
            allow_from_causal = make_causal_allow_mask(t, s, device).expand(bsz, 1, -1, -1)

        return combine_allow_masks(allow_from_causal, allow_from_padding, allow_from_attention)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        causal: bool = True,
        past_key_values: PastKeyValues = None,
        use_cache: bool = False,
    ):
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be (B, T).")
        bsz, t = input_ids.shape

        allow_mask = self._build_allow_mask(input_ids, attention_mask, key_padding_mask, causal, past_key_values)

        # seq offset for RoPE under cache
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_k, _ = past_key_values[0]
            seq_offset = past_k.size(2)
        else:
            seq_offset = 0

        x = self.embed(input_ids)  # (B,T,D)

        present: List[KVCache] = []
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, pkv = layer(x, allow_mask=allow_mask, past_kv=layer_past, use_cache=use_cache, seq_offset=seq_offset)
            if use_cache:
                if pkv is None:
                    raise RuntimeError("use_cache=True but no pkv returned.")
                present.append(pkv)

        if self.final_norm is not None:
            x = self.final_norm(x)

        logits = self.lm_head(x)

        return (logits, present) if use_cache else logits

    @torch.no_grad()
    def greedy_generate(self, prompt_ids: Tensor, max_new_tokens: int = 64, eos_token_id: Optional[int] = None) -> Tensor:
        self.eval()
        out = prompt_ids
        past: List[KVCache] = [None] * self.cfg.num_layers  # type: ignore

        # Build cache on full prompt
        logits, past = self(out, causal=True, use_cache=True, past_key_values=None)

        for _ in range(max_new_tokens):
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
            # Decode with cache (no causal needed because we only append new keys)
            logits, past = self(next_id, causal=False, use_cache=True, past_key_values=past)

        return out


# -----------------------------
# Loading weights (ModelPack)
# -----------------------------

def load_modelpack_from_json(path: str) -> ModelPack:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ModelPack(**data)


def _load_state_dict_safetensors(shard_paths: List[str], device: Optional[Union[str, torch.device]] = None) -> Dict[str, Tensor]:
    """
    Loads safetensors shards and merges into a single state_dict.
    Requires: pip install safetensors
    """
    from safetensors.torch import load_file  # type: ignore
    sd: Dict[str, Tensor] = {}
    for p in shard_paths:
        part = load_file(p, device=str(device) if device is not None else "cpu")
        sd.update(part)
    return sd


def load_from_modelpack_local(manifest_path: str, shard_dir: str, device: Optional[Union[str, torch.device]] = None) -> ProductionTransformerRAGE:
    """
    Load a model from a local ModelPack manifest + local shard files.
    This keeps the "IPFS distribution" concern outside the model code: IPFS fetchers
    should put shards into shard_dir using filenames in the manifest.
    """
    mp = load_modelpack_from_json(manifest_path)
    cfg = RAGETransformerConfig(**{k: mp.config[k] for k in mp.config})
    model = ProductionTransformerRAGE(cfg)

    shard_paths = [os.path.join(shard_dir, s["filename"]) for s in mp.shards]
    if mp.format.lower() == "safetensors":
        state = _load_state_dict_safetensors(shard_paths, device=device)
    else:
        raise ValueError("Unsupported format. Use safetensors for production safety.")

    model.load_state_dict(state, strict=True)
    if device is not None:
        model.to(device)
    model.eval()
    return model
'''
code_path = Path("/mnt/data/production_transformer_rage_v1.1.0.py")
code_path.write_text(code, encoding="utf-8")

# 2) Create a packaging spec for IPFS "ModelPack" + best practices for scalability
pack_spec = f"""# RAGE ModelPack · IPFS-Native Distribution Spec (v1.0)

**Last updated:** {date(2026,2,14).isoformat()}

This spec defines how to distribute **RAGE generation artifacts** (models, adapters, and metadata) using **IPFS content addressing** for *universal access* and *horizontal scalability*.

**Principle:** store *artifacts* by CID (integrity), and store *pointers* (latest/curated) via a mutable layer (IPNS/ENS/DNS), because CIDs are immutable. citeturn0search2turn0search19

---

## 1) Goals

### Universal access
- Any node (desktop / VPS / edge / CI) can fetch an artifact from IPFS via:
  - local IPFS daemon
  - gateway
  - mirrored storage
- Same artifact identifier everywhere: **CID**.

### Infinite scalability (practical meaning)
- **Content-addressing** enables:
  - global caching and deduplication
  - shard-level reuse (only download what you need)
  - immutable versioning (every build produces a new CID)
- Scale comes from distribution + caching, not from one server.

---

## 2) Best-practice artifact formats

### 2.1 Prefer `safetensors` for weights
`SafeTensors` is a safe, fast, zero-copy tensor container that avoids pickle deserialization risks. citeturn0search1turn0search5turn0search12

**Recommendation**
- Store weights as **sharded safetensors** files: `model-00001-of-000xx.safetensors`.
- Pin shards to IPFS; record each shard CID.

### 2.2 Always publish a manifest
The manifest is a small JSON object (also stored on IPFS). The manifest CID becomes the **model version identifier**.

---

## 3) ModelPack Manifest

### 3.1 Minimal schema (JSON)
```json
{{
  "format": "safetensors",
  "model_name": "ProductionTransformer-RAGE",
  "model_version": "1.1.0",
  "config": {{
    "vocab_size": 32000,
    "d_model": 512,
    "num_layers": 8,
    "num_heads": 8,
    "num_kv_heads": 2,
    "dim_ff": 2048,
    "dropout": 0.0,
    "max_seq_len": 4096,
    "rope_theta": 10000.0,
    "tie_weights": true
  }},
  "shards": [
    {{
      "filename": "model-00001-of-00004.safetensors",
      "cid": "bafy...",
      "sha256": "sha256:..."
    }}
  ],
  "tokenizer": {{
    "type": "bpe|sentencepiece|...",
    "cid": "bafy..."
  }},
  "license": "MIT|Apache-2.0|...",
  "provenance": {{
    "build_ts": "2026-02-14T00:00:00Z",
    "git_commit": "abc123",
    "trainer_wallet": "0x...",
    "signature": "..."
  }}
}}
