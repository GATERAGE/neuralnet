# SimpleMind in RAGE neuralnet

SimpleMind is useful in `neuralnet` **if you treat it as a policy/reranking brain**, not as a generator.

## What it does best
- **Reranking** retrieved chunks (capsules) before building the prompt
- **Trust gating**: prefer CID+signature+validated sources by default
- **Routing**: local vs remote LLM, or “promote to permanence” decisions

## Where it sits in the RAG pipeline

1. Retrieve candidates (FAISS semantic + optional lexical)
2. Compute feature vectors per candidate
3. Score candidates with SimpleMind
4. Select top-K, build context, generate response
5. Log: retrieval set + scores + chosen context (logs are memory)

## Recommended implementation choice
### For neuralnet (PyTorch repo)
Use **`simplemind_torch.py`**:
- keeps dependencies consistent (PyTorch already required)
- deploys everywhere easily

### If you specifically want JAX for training
Use **`simplemind_jax.py`** for offline training, then export weights and port them.

## Feature schema (example)
A simple candidate feature vector might look like:

| Feature | Type | Meaning |
|---|---|---|
| `vec_sim` | float | cosine/inner-product similarity from FAISS |
| `lex_score` | float | BM25 / keyword score (0 if unused) |
| `integrity_level` | float | L0-L5 normalized (0..1) |
| `signed` | float | 1 if signature receipt present |
| `validated` | float | 1 if quorum validated |
| `freshness_days` | float | age in days (use log1p scaling) |
| `source_rep` | float | reputation score (0..1) |
| `len_tokens` | float | chunk length normalized |

**Best practice:** version this schema as a JSON and pin it (CID).  
The reranker ModelPack should reference `feature_schema_cid`.

## Minimal integration sketch (rag_inference.py)

After you retrieve `top_k` chunks, compute features and rerank:

```python
# pseudo-code
features = build_features(candidates)             # shape (K, D)
probs = simplemind.predict_proba(features)        # shape (K, 1)
reranked = sort_by(probs, candidates)
context = join(top_n(reranked))
```

## What you gain
- Better grounding: less irrelevant context in the prompt
- Better trust: default to verified memory
- Better cost: fewer tokens wasted → faster inference
