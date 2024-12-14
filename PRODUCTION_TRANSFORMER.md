Production_Transformer.py: A Minimal Local Transformer Model

This project implements a minimal Transformer architecture designed for sequence modeling tasks like language modeling. It includes key components such as Positional Encoding, Multi-Head Self-Attention, and Transformer Encoder Blocks, all built using PyTorch.
Features

    Positional Encoding: Injects positional information into token embeddings.
    Multi-Head Self-Attention: Allows the model to focus on different parts of the sequence simultaneously.
    Layer Normalization: Improves training stability and convergence.
    Feedforward Layers: Enhances the model's representational power.
    Causal Masking: Enables autoregressive decoding for tasks like text generation.
    Weight Sharing: Optionally ties the embedding and output layers for efficiency.

File Overview
Key Classes

    PositionalEncoding
        Injects positional information into token embeddings using sine and cosine functions.
        Accounts for sequence length (max_len) and model dimensionality (d_model).

    MultiHeadSelfAttention
        Implements scaled dot-product attention with multiple attention heads.
        Handles optional masking for causal decoding or padding.

    TransformerBlock
        Represents a single Transformer encoder block with:
            Multi-Head Self-Attention.
            Feedforward layers.
            Layer normalization and residual connections.

    ProductionTransformer
        Combines embeddings, positional encodings, and multiple Transformer blocks.
        Outputs logits for language modeling tasks.
        Includes optional weight tying between the embedding and output layers.

    create_causal_mask
        Generates a triangular causal mask for autoregressive decoding.

Model Initialization

The ProductionTransformer class supports flexible initialization with customizable parameters:
```bash
    vocab_size: Vocabulary size for token embeddings.
    d_model: Dimensionality of the model.
    num_heads: Number of attention heads.
    num_layers: Number of Transformer layers.
    dim_feedforward: Dimensionality of feedforward layers.
    dropout: Dropout rate.
    max_len: Maximum sequence length.
    tie_weights: Whether to tie the embedding and output layers.
```
Forward Pass

The forward method expects:

    Input tensor x of shape (batch_size, seq_len) containing token indices.
    Optional mask tensor mask of shape (batch_size, seq_len, seq_len) for padding or causal masking.

It outputs:

    Logits of shape (batch_size, seq_len, vocab_size).

Setup

    Install dependencies:

```bash
pip install torch
``
Import and initialize the model:
```bash
    import torch
    from your_module import ProductionTransformer, create_causal_mask

    model = ProductionTransformer(vocab_size=10000, d_model=128, num_heads=4, num_layers=2)
```
Example
```bash
# Sample input: Batch of tokenized sequences (batch_size=2, seq_len=5)
x = torch.randint(0, 10000, (2, 5))

# Create a causal mask
mask = create_causal_mask(seq_len=x.size(1))

# Forward pass
logits = model(x, mask=mask)

# Output shape: (batch_size=2, seq_len=5, vocab_size=10000)
print(logits.shape)
```
Explanation of Components

    Positional Encoding:
        Adds sinusoidal positional information to token embeddings, enabling the model to distinguish token positions.

    Multi-Head Self-Attention:
        Splits input into multiple heads for parallel attention computation.
        Computes scaled dot-product attention and combines the outputs.

    Transformer Block:
        Combines self-attention and feedforward layers with residual connections and layer normalization.

    Causal Mask:
        Masks future tokens to ensure predictions depend only on past tokens.

Requirements

    Python 3.10
    PyTorch 1.7+
