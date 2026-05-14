#!/usr/bin/env python3
# production_transformer.py
"""
ProductionTransformer v1.0.0
A compact, decoder-style Transformer for language modeling logits.

Goals (v1.0.0)
- Pre-norm Transformer blocks (better stability than post-norm as depth grows)
- Safer masking (bool "allow" masks; dtype-safe -inf handling)
- Optional padding mask + causal mask composition
- Optional KV-cache for autoregressive decoding
- Optional use of PyTorch SDPA (scaled_dot_product_attention) when available
- Clear, typed interfaces and practical helpers

Non-goals
- Tokenizer / BPE
- Full-feature generation (sampling strategies, beam search, logits processors)
- Distributed training, quantization, LoRA, etc.

Mask semantics used by this module
- `attn_mask` is a **boolean allow mask**: True = token is allowed to attend; False = masked.
  This matches PyTorch SDPA documentation for boolean masks.
- `key_padding_mask` is a **boolean padding mask**: True = padding (masked out), False = real token.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = "1.0.0"

Tensor = torch.Tensor
KVCache = Tuple[Tensor, Tensor]  # (k, v) with shape (B, H, S, d_k)
PastKeyValues = Optional[List[KVCache]]


@dataclass(frozen=True)
class ProductionTransformerConfig:
    vocab_size: int = 10000
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 2048
    tie_weights: bool = True
    use_sdpa: bool = True  # use torch.nn.functional.scaled_dot_product_attention when available
    activation: str = "gelu"  # "gelu" or "relu"
    final_layer_norm: bool = True


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (non-trainable)."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D)
        t = x.size(1)
        if t > self.pe.size(1):
            raise ValueError(f"seq_len={t} exceeds max_len={self.pe.size(1)}; increase max_len.")
        return x + self.pe[:, :t, :]


def create_causal_allow_mask(seq_len_q: int, seq_len_k: Optional[int] = None, device: Union[str, torch.device] = "cpu") -> Tensor:
    """
    Create a boolean **allow** mask for causal attention.
    True = allowed to attend, False = masked.

    Returns shape: (1, 1, L, S) where L=seq_len_q and S=seq_len_k (defaults to L).
    """
    if seq_len_k is None:
        seq_len_k = seq_len_q
    # For standard self-attention without cache, L == S and tril is correct.
    # For cached decoding (L may be 1 and S > 1), causal masking is not required (no future keys exist),
    # so callers should set causal=False and/or pass an explicit mask.
    mask = torch.ones((seq_len_q, seq_len_k), dtype=torch.bool, device=device)
    mask = torch.tril(mask, diagonal=0)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, S)


def _key_padding_to_allow_mask(key_padding_mask: Tensor, seq_len_q: int) -> Tensor:
    """
    Convert key_padding_mask (B, S) where True=pad into an allow mask (B, 1, L, S) where True=allowed.
    """
    if key_padding_mask.dtype != torch.bool:
        key_padding_mask = key_padding_mask.bool()
    allow_keys = ~key_padding_mask  # True where real tokens
    # (B, 1, 1, S) -> (B, 1, L, S)
    return allow_keys[:, None, None, :].expand(-1, 1, seq_len_q, -1)


def _normalize_attention_mask(
    attention_mask: Optional[Tensor],
    batch_size: int,
    seq_len_q: int,
    seq_len_k: int,
    device: torch.device,
) -> Optional[Tensor]:
    """
    Normalize attention_mask to a boolean allow mask broadcastable to (B, 1, L, S).

    Supported inputs:
    - None
    - (B, S) token mask: 1/True = keep, 0/False = pad
    - (L, S) allow mask
    - (B, 1, L, S) allow mask
    """
    if attention_mask is None:
        return None

    if attention_mask.dtype != torch.bool and attention_mask.dtype != torch.float32 and attention_mask.dtype != torch.float16 and attention_mask.dtype != torch.bfloat16 and attention_mask.dtype != torch.int64 and attention_mask.dtype != torch.int32:
        attention_mask = attention_mask.to(torch.bool)

    # (B, S) -> interpret as keep-mask (True/1 = keep)
    if attention_mask.dim() == 2 and attention_mask.size(0) == batch_size and attention_mask.size(1) == seq_len_k:
        keep = attention_mask.bool()
        # Convert keep->padding then to allow-mask
        key_padding_mask = ~keep  # True where pad
        return _key_padding_to_allow_mask(key_padding_mask, seq_len_q)

    # (L, S) -> expand to (1,1,L,S) then broadcast across batch
    if attention_mask.dim() == 2 and attention_mask.size(0) == seq_len_q and attention_mask.size(1) == seq_len_k:
        return attention_mask.bool().to(device).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    # (B, 1, L, S)
    if attention_mask.dim() == 4 and attention_mask.size(0) == batch_size and attention_mask.size(2) == seq_len_q and attention_mask.size(3) == seq_len_k:
        return attention_mask.bool().to(device)

    raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")


def _combine_allow_masks(*masks: Optional[Tensor]) -> Optional[Tensor]:
    """Logical-AND combine boolean allow masks; ignores None."""
    out: Optional[Tensor] = None
    for m in masks:
        if m is None:
            continue
        out = m if out is None else (out & m)
    return out


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with optional SDPA and optional KV-cache.

    Inputs:
      x: (B, T, D)
      allow_mask: bool allow mask broadcastable to (B, 1, T, S) where S is key length
      past_kv: optional cached (k, v) each (B, H, S_past, d_k)
      use_cache: whether to return present_kv
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, use_sdpa: bool = True):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_sdpa = use_sdpa and hasattr(F, "scaled_dot_product_attention")

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _shape(self, x: Tensor, bsz: int) -> Tensor:
        # (B, T, D) -> (B, H, T, d_k)
        return x.view(bsz, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        allow_mask: Optional[Tensor] = None,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        bsz, t, _ = x.size()

        q = self._shape(self.q_proj(x), bsz)  # (B, H, T, d_k)
        k = self._shape(self.k_proj(x), bsz)
        v = self._shape(self.v_proj(x), bsz)

        if past_kv is not None:
            past_k, past_v = past_kv
            # concat on sequence dimension
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        # allow_mask should be broadcastable to (B, H, T, S)
        if allow_mask is not None:
            # convert (B,1,T,S) -> broadcast to heads
            if allow_mask.dim() == 4 and allow_mask.size(1) == 1:
                allow_mask = allow_mask.expand(bsz, self.num_heads, allow_mask.size(2), allow_mask.size(3))

        if self.use_sdpa:
            # SDPA boolean attn_mask semantics: True => participates in attention (allowed).
            # dropout_p must be set to 0.0 when not training.
            dropout_p = self.dropout if self.training else 0.0
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow_mask, dropout_p=dropout_p, is_causal=False)
        else:
            # Manual attention (stable, explicit)
            s = k.size(2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, T, S)

            if allow_mask is not None:
                # allow_mask: True allowed, False masked
                min_val = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(~allow_mask, min_val)

            attn = torch.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            attn_out = torch.matmul(attn, v)  # (B, H, T, d_k)

        # (B, H, T, d_k) -> (B, T, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, t, self.d_model)
        out = self.out_proj(attn_out)
        return out, present_kv


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        act = activation.lower()
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError("activation must be 'gelu' or 'relu'")

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-norm decoder block:
      x = x + dropout(Attn(LN(x)))
      x = x + dropout(FFN(LN(x)))
    """

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float, use_sdpa: bool, activation: str):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout=dropout, use_sdpa=use_sdpa)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dim_feedforward, dropout=dropout, activation=activation)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        allow_mask: Optional[Tensor] = None,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        # Attention
        a, present_kv = self.attn(self.ln1(x), allow_mask=allow_mask, past_kv=past_kv, use_cache=use_cache)
        x = x + self.drop1(a)

        # FFN
        f = self.ff(self.ln2(x))
        x = x + self.drop2(f)
        return x, present_kv


class ProductionTransformer(nn.Module):
    """
    Decoder-style Transformer that returns logits for each token position.

    Inputs
      input_ids: (B, T) token indices
      attention_mask: optional; supports:
        - (B, T) keep-mask (1/True = keep, 0/False = pad)
        - (T, T) allow mask
        - (B, 1, T, T) allow mask
      key_padding_mask: optional (B, T) where True = pad
      causal: whether to apply causal masking (recommended for LM training)
      past_key_values: optional list of (k,v) caches per layer
      use_cache: return present_key_values for decoding

    Returns
      logits: (B, T, vocab_size)
      present_key_values (optional): list of (k,v) per layer when use_cache=True
    """

    def __init__(self, config: Optional[ProductionTransformerConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = ProductionTransformerConfig(**kwargs)

        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_len)
        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    use_sdpa=config.use_sdpa,
                    activation=config.activation,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_ln = nn.LayerNorm(config.d_model) if config.final_layer_norm else None
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier init for stability; embeddings are initialized by nn.Embedding by default.
        for name, p in self.named_parameters():
            if p.dim() > 1 and "embedding" not in name:
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
        bsz, t = input_ids.size()

        # Determine key length (S). With KV cache, S grows beyond T.
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_k, _ = past_key_values[0]
            s = past_k.size(2) + t
        else:
            s = t

        # Normalize attention_mask to allow-mask (B,1,L,S) if provided.
        # If attention_mask is (B,T) keep mask, it will be treated as padding and expanded to keys.
        allow_from_attention_mask = _normalize_attention_mask(attention_mask, bsz, t, s if attention_mask is None or attention_mask.dim() != 2 else t, device)

        # key_padding_mask masks KEYS (shape B,S). We only have current T padding info; when using cache
        # you must manage padding yourself. Typical generation uses no padding.
        allow_from_padding = None
        if key_padding_mask is not None:
            if key_padding_mask.size(0) != bsz:
                raise ValueError("key_padding_mask batch mismatch.")
            if key_padding_mask.size(1) != t:
                raise ValueError("key_padding_mask must match current sequence length (B,T).")
            allow_from_padding = _key_padding_to_allow_mask(key_padding_mask, seq_len_q=t)

        # Causal allow mask:
        allow_from_causal = None
        # For cached decoding with q_len=1, causal mask is not needed (no future keys exist).
        if causal and not (past_key_values is not None and t == 1 and s > 1):
            allow_from_causal = create_causal_allow_mask(seq_len_q=t, seq_len_k=s, device=device).expand(bsz, 1, -1, -1)

        return _combine_allow_masks(allow_from_causal, allow_from_padding, allow_from_attention_mask)

    @torch.no_grad()
    def greedy_generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 32,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Minimal greedy decode using KV cache.
        - prompt_ids: (B, T)
        - returns: (B, T + max_new_tokens) (or earlier if eos_token_id hit for all batches)
        """
        self.eval()
        out = prompt_ids
        past: List[KVCache] = [None] * self.num_layers  # type: ignore

        # First pass: build cache from full prompt
        logits, past = self(out, causal=True, use_cache=True, past_key_values=None)
        # Now decode one token at a time
        for _ in range(max_new_tokens):
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)

            if eos_token_id is not None:
                if (next_id == eos_token_id).all():
                    break

            # Only feed the latest token; cache carries the context
            logits, past = self(next_id, causal=False, use_cache=True, past_key_values=past)

        return out

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        causal: bool = True,
        use_cache: bool = False,
        past_key_values: PastKeyValues = None,
    ) -> Union[Tensor, Tuple[Tensor, List[KVCache]]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be (B, T) token indices.")
        bsz, t = input_ids.size()

        allow_mask = self._build_allow_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            causal=causal,
            past_key_values=past_key_values,
        )

        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.drop(x)

        present: List[KVCache] = []
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, pkv = layer(x, allow_mask=allow_mask, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                if pkv is None:
                    raise RuntimeError("use_cache=True but layer did not return pkv.")
                present.append(pkv)

        if self.final_ln is not None:
            x = self.final_ln(x)

        logits = self.lm_head(x)

        if use_cache:
            return logits, present
        return logits
