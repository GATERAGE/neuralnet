# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the neuralnet package (v0.1.0a4).

Each heavy-dep test is gated by `pytest.importorskip(...)` — runs when the
dependency is installed, skips cleanly when absent. The smoke tests
(tests/test_smoke.py) stay no-dep so the basic contract is always checked.

Heavy paths covered:
  - ProductionTransformer forward (torch)
  - ProductionTransformerRAGE forward (torch + the RAGE-flavored config)
  - SimpleMind reranker forward (torch)
  - LLMRouter local-mode construction (torch + openai for the import)

Pure-Python paths covered (no skip needed):
  - RAGEDataLoader chunkification
  - ModelPack template generation

Add the repo root to sys.path so the canonical `neuralnet.*` imports work
when pytest is run from inside the cloned repo without `pip install -e .`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ───────────────────────────────────────────────────────────────────────────
# Pure-Python tests (no skip needed)
# ───────────────────────────────────────────────────────────────────────────


def test_dataloader_chunk_text_word_windows():
    """RAGEDataLoader._chunk_text produces fixed-size word-window chunks."""
    from neuralnet.dataloader import RAGEDataLoader
    loader = RAGEDataLoader(chunk_size=4)
    text = "one two three four five six seven eight"
    chunks = loader._chunk_text(text)
    # 8 words / 4-word window = 2 non-overlapping chunks.
    assert len(chunks) == 2
    assert chunks[0] == "one two three four"
    assert chunks[1] == "five six seven eight"


def test_dataloader_chunk_text_short_text():
    """Text shorter than chunk_size yields one chunk."""
    from neuralnet.dataloader import RAGEDataLoader
    loader = RAGEDataLoader(chunk_size=100)
    chunks = loader._chunk_text("hello world")
    assert len(chunks) == 1
    assert chunks[0] == "hello world"


def test_modelpack_init_template_runs_via_main():
    """`python -m neuralnet.modelpack init-template --out <path>` writes a JSON."""
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = tmp.name
    try:
        result = subprocess.run(
            [sys.executable, "-m", "neuralnet.modelpack", "init-template", "--out", out_path],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"init-template failed: {result.stderr}"
        # The JSON should parse and have the manifest shape.
        data = json.loads(Path(out_path).read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        # Spec docs say the manifest carries shards + config; the exact keys
        # depend on the embedded template. At minimum a non-empty dict.
        assert len(data) > 0
    finally:
        Path(out_path).unlink(missing_ok=True)


# ───────────────────────────────────────────────────────────────────────────
# Torch-gated tests
# ───────────────────────────────────────────────────────────────────────────


def test_production_transformer_forward():
    """neuralnet.transformer.ProductionTransformer runs a forward pass."""
    torch = pytest.importorskip("torch")
    from neuralnet.transformer import ProductionTransformer

    vocab_size = 128
    seq_len = 16
    batch = 2

    model = ProductionTransformer(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        num_layers=2,
        dim_feedforward=64,
        max_len=seq_len,
        dropout=0.0,
    )
    model.eval()
    tokens = torch.randint(0, vocab_size, (batch, seq_len))
    with torch.no_grad():
        logits = model(tokens)
    # Shape: (batch, seq_len, vocab_size)
    assert logits.shape == (batch, seq_len, vocab_size)
    assert logits.dtype == torch.float32


def test_production_transformer_causal_mask():
    """create_causal_mask returns a triangular allow-mask of the right shape."""
    torch = pytest.importorskip("torch")
    from neuralnet.transformer import create_causal_mask

    mask = create_causal_mask(5)
    assert mask.shape[-2:] == (5, 5)
    # Lower-triangular structure (position i can attend to j<=i).
    # Implementation may return bool, int, or float — check the pattern.
    m = mask.float().squeeze().squeeze() if mask.dim() > 2 else mask.float()
    # Diagonal + below should be one of {1, True}; above should be 0/False or -inf.
    # We check the upper-triangle (excluding diagonal) is "blocked".
    for i in range(5):
        for j in range(i + 1, 5):
            # masked-out entries are either 0 (boolean allow-mask) or -inf
            val = float(m[i, j])
            assert val == 0.0 or val == float("-inf"), (
                f"upper-triangle at ({i},{j}) should be blocked, got {val}"
            )


def test_production_transformer_rage_forward():
    """neuralnet.transformer_rage.ProductionTransformerRAGE forward pass."""
    torch = pytest.importorskip("torch")
    from neuralnet.transformer_rage import (
        ProductionTransformerRAGE,
        RAGETransformerConfig,
    )

    cfg = RAGETransformerConfig(
        vocab_size=256,
        d_model=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,    # GQA
        dim_ff=128,
        dropout=0.0,
        max_seq_len=32,
        rope_theta=10000.0,
        tie_weights=True,
    )
    model = ProductionTransformerRAGE(cfg)
    model.eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        out = model(tokens)
    # Output may be a tensor or a (logits, ...) tuple depending on KV-cache mode.
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (1, 16, cfg.vocab_size)


def test_simplemind_torch_forward():
    """neuralnet.simplemind reranker forward."""
    torch = pytest.importorskip("torch")
    from neuralnet.simplemind import SimpleMindTorch, SimpleMindTorchConfig

    cfg = SimpleMindTorchConfig(input_size=8, hidden_sizes=(16, 8), output_size=1)
    model = SimpleMindTorch(cfg)
    model.eval()
    x = torch.randn(4, 8)  # batch of 4 candidate feature-vectors
    with torch.no_grad():
        out = model(x)
    # Reranker returns logits or (logits, probs) — be permissive.
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape[0] == 4


def test_llm_router_constructs():
    """LLMRouter constructs without error (no inference)."""
    pytest.importorskip("openai")  # router imports openai at module load
    pytest.importorskip("torch")
    from neuralnet.router import LLMRouter

    router = LLMRouter(local_vocab_size=128)
    # Sanity: env defaults wired up
    assert router.ollama_endpoint.endswith(":11434"), (
        f"Ollama endpoint should default to :11434, got {router.ollama_endpoint}"
    )
