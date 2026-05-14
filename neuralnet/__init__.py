# SPDX-License-Identifier: Apache-2.0
# (c) 2024-2026 GATERAGE — neuralnet
"""
neuralnet — RAGE training + serving substrate.

Fourth corner of the GATERAGE architecture:
    RAGE remembers, aGLM decides, MASTERMIND orchestrates,
    neuralnet trains and serves.

⚠️  PROTOTYPE — version 0.1.0a3 (PEP-440 alpha)
Interfaces will change. Pin the commit SHA in production.

Public modules:
    neuralnet.transformer        — teaching minimal transformer (127 LOC)
    neuralnet.transformer_v1     — pre-norm + KV cache + SDPA cleaned-up version
    neuralnet.transformer_rage   — RAGE-flavored v1.1 (RMSNorm + SwiGLU + GQA + RoPE + ModelPack)
    neuralnet.router             — LLMRouter (local / openai / together / ollama)
    neuralnet.inference          — RAGInference orchestrator (FAISS + retrieval + LLMRouter)
    neuralnet.dataloader         — RAGEDataLoader (txt/md/pdf/docx/URL chunking)
    neuralnet.simplemind         — SimpleMind reranker (PyTorch)
    neuralnet.modelpack          — IPFS ModelPack fetch CLI + lib

Public symbols re-exported here for convenience:
    ProductionTransformer            (from .transformer)
    ProductionTransformerRAGE        (from .transformer_rage)
    RAGETransformerConfig            (from .transformer_rage)
    ModelPack                        (from .transformer_rage)
    LLMRouter                        (from .router)
    RAGInference                     (from .inference)
    RAGEDataLoader                   (from .dataloader)
"""
from __future__ import annotations

__version__ = "0.1.0a3"

# ─── public top-level re-exports ────────────────────────────────────────
# Heavy modules are imported lazily inside try/except so that
# `import neuralnet` succeeds even when torch / faiss / sentence-transformers
# are not installed. Consumers who need a specific class import the module
# directly.

__all__ = ["__version__"]

try:
    from .transformer import ProductionTransformer  # type: ignore[F401]
    __all__.append("ProductionTransformer")
except ImportError:
    pass

try:
    from .transformer_rage import (  # type: ignore[F401]
        ProductionTransformerRAGE,
        RAGETransformerConfig,
        ModelPack,
        load_from_modelpack_local,
    )
    __all__ += ["ProductionTransformerRAGE", "RAGETransformerConfig",
                "ModelPack", "load_from_modelpack_local"]
except ImportError:
    pass

try:
    from .router import LLMRouter  # type: ignore[F401]
    __all__.append("LLMRouter")
except ImportError:
    pass

try:
    from .inference import RAGInference  # type: ignore[F401]
    __all__.append("RAGInference")
except ImportError:
    pass

try:
    from .dataloader import RAGEDataLoader  # type: ignore[F401]
    __all__.append("RAGEDataLoader")
except ImportError:
    pass
