#!/usr/bin/env python3
"""
generate.py
RAGE utility: run local generation with ProductionTransformerRAGE using KV cache + sampling.

Tokenizer-agnostic by default:
- pass `--prompt-ids '[1,2,3]'` (JSON list of ints)
- OR pass `--tokenizer <hf_name_or_path>` and `--prompt "..."` to use AutoTokenizer

Best practice for RAGE
- Stamp outputs with: model manifest path/CID, decoding params, timestamp.
- Treat the run log as "memory" (append-only event).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

import torch

# Local import: keep in same folder or adjust PYTHONPATH
from production_transformer_rage_v1.1.0 import load_from_modelpack_local, ProductionTransformerRAGE  # type: ignore


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, top_k, dim=-1).values[..., -1].unsqueeze(-1)
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)

        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    return logits


def apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    bsz = logits.size(0)
    for b in range(bsz):
        seen = torch.unique(generated[b])
        for tok in seen:
            t = int(tok.item())
            v = logits[b, t]
            logits[b, t] = torch.where(v > 0, v / penalty, v * penalty)
    return logits


@torch.no_grad()
def sample_generate(
    model: ProductionTransformerRAGE,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    eos_token_id: Optional[int],
) -> torch.Tensor:
    model.eval()
    out = prompt_ids
    logits, past = model(out, causal=True, use_cache=True, past_key_values=None)

    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :]

        if temperature <= 0:
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            next_logits = apply_repetition_penalty(next_logits, out, repetition_penalty)
            next_logits = top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        out = torch.cat([out, next_id], dim=1)

        if eos_token_id is not None and (next_id == eos_token_id).all():
            break

        logits, past = model(next_id, causal=False, use_cache=True, past_key_values=past)

    return out


def maybe_load_tokenizer(tokenizer_ref: Optional[str]):
    if not tokenizer_ref:
        return None
    from transformers import AutoTokenizer  # type: ignore
    return AutoTokenizer.from_pretrained(tokenizer_ref, use_fast=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="RAGE local generation (ProductionTransformerRAGE)")
    ap.add_argument("--manifest", required=True, help="Path to ModelPack manifest JSON (local file)")
    ap.add_argument("--shard-dir", required=True, help="Directory containing shard files referenced by the manifest")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps (if available)")
    ap.add_argument("--tokenizer", default=None, help="Optional: HF tokenizer name/path for encode/decode")

    ap.add_argument("--prompt", default=None, help="Prompt string (requires --tokenizer)")
    ap.add_argument("--prompt-ids", default=None, help="JSON list of token ids (no tokenizer required)")

    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--eos", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--log-out", default=None, help="Optional: write a JSON run log to this path")
    args = ap.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model = load_from_modelpack_local(manifest_path=args.manifest, shard_dir=args.shard_dir, device=device)

    tok = maybe_load_tokenizer(args.tokenizer)

    if args.prompt_ids:
        ids = json.loads(args.prompt_ids)
        if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
            raise SystemExit("--prompt-ids must be a JSON list of integers")
        prompt_ids = torch.tensor([ids], dtype=torch.long, device=device)
    elif args.prompt:
        if tok is None:
            raise SystemExit("--prompt requires --tokenizer")
        enc = tok(args.prompt, return_tensors="pt")
        prompt_ids = enc["input_ids"].to(device)
    else:
        raise SystemExit("Provide either --prompt-ids or --prompt")

    t0 = time.time()
    out_ids = sample_generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=args.eos,
    )
    dt = time.time() - t0

    if tok is not None:
        text = tok.decode(out_ids[0].tolist(), skip_special_tokens=True)
        print(text)
    else:
        print(json.dumps({"output_ids": out_ids[0].tolist()}, indent=2))

    if args.log_out:
        run: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": str(device),
            "manifest_path": os.path.abspath(args.manifest),
            "shard_dir": os.path.abspath(args.shard_dir),
            "decode": {
                "max_new": args.max_new,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "eos": args.eos,
                "seed": args.seed,
            },
            "latency_s": dt,
            "prompt_len": int(prompt_ids.size(1)),
            "output_len": int(out_ids.size(1)),
            "output_ids": out_ids[0].tolist(),
        }
        os.makedirs(os.path.dirname(args.log_out) or ".", exist_ok=True)
        with open(args.log_out, "w", encoding="utf-8") as f:
            json.dump(run, f, indent=2)
        print(f"\nWrote log: {args.log_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
