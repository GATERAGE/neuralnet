#!/usr/bin/env python3
# simplemind_torch.py
"""
SimpleMind (PyTorch) — RAGE-friendly policy/reranking MLP.

Why Torch for neuralnet
- neuralnet already uses PyTorch (production_transformer, training scaffolds)
- avoids adding JAX/Optax as a second framework
- deploys easily on CPU/VPS/edge with standard wheels

Typical RAGE usage
- Train offline on feature vectors: X = [vector_sim, bm25, integrity, freshness, ...]
- Deploy as a tiny reranker: score candidates and choose top-K for context

Outputs
- logits and probabilities (sigmoid), plus convenience metrics.

License: follow repo license.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SimpleMindTorchConfig:
    input_size: int
    hidden_sizes: Tuple[int, ...] = (64, 32)
    output_size: int = 1
    activation: str = "relu"  # relu|gelu|tanh|sigmoid|leaky_relu
    dropout: float = 0.0


def _activation(name: str) -> nn.Module:
    n = name.lower()
    if n == "relu":
        return nn.ReLU()
    if n == "gelu":
        return nn.GELU()
    if n == "tanh":
        return nn.Tanh()
    if n == "sigmoid":
        return nn.Sigmoid()
    if n == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError("activation must be relu|gelu|tanh|sigmoid|leaky_relu")


class SimpleMindTorch(nn.Module):
    def __init__(self, cfg: SimpleMindTorchConfig):
        super().__init__()
        self.cfg = cfg
        act = _activation(cfg.activation)

        layers: List[nn.Module] = []
        in_dim = cfg.input_size
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act)
            if cfg.dropout and cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns logits
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def loss_bce_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.float()
        if y.dim() == 1:
            y = y.view(-1, 1)
        return F.binary_cross_entropy_with_logits(logits, y)

    @staticmethod
    @torch.no_grad()
    def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        if y.dim() == 1:
            y = y.view(-1, 1)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).to(torch.int32)
        y_i = y.to(torch.int32)

        acc = (pred == y_i).float().mean().item()

        tp = ((pred == 1) & (y_i == 1)).sum().item()
        fp = ((pred == 1) & (y_i == 0)).sum().item()
        fn = ((pred == 0) & (y_i == 1)).sum().item()

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

        return {"accuracy": acc, "precision": float(precision), "recall": float(recall), "f1_score": float(f1)}


def train_epoch(
    model: SimpleMindTorch,
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = model.loss_bce_logits(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: SimpleMindTorch,
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    losses = []
    all_logits = []
    all_y = []
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = model.loss_bce_logits(logits, y)
        losses.append(float(loss.item()))
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
    if not losses:
        return {}
    logits_cat = torch.cat(all_logits, dim=0)
    y_cat = torch.cat(all_y, dim=0)
    m = model.metrics_from_logits(logits_cat, y_cat)
    m["loss"] = sum(losses) / len(losses)
    return m
