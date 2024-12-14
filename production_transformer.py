#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Inject positional information into token embeddings."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional masking."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out   = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        bsz, seq_len, d_model = x.size()
        Q = self.query(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.out(context)

class TransformerBlock(nn.Module):
    """One transformer encoder block."""
    def __init__(self, d_model, num_heads, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x

class ProductionTransformer(nn.Module):
    """
    Minimal local Transformer model for sequence tasks, e.g. language modeling.
    """
    def __init__(self, vocab_size=10000, d_model=128, num_heads=4, num_layers=2,
                 dim_feedforward=512, dropout=0.1, max_len=512, tie_weights=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        """
        x shape: (batch_size, seq_len)
        mask shape: (batch_size, seq_len, seq_len) or None
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        logits = self.lm_head(x)
        return logits

def create_causal_mask(seq_len, device='cpu'):
    """
    Triangular causal mask for autoregressive decoding.
    shape: (1, seq_len, seq_len)
    """
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)
