from pathlib import Path
import json
from datetime import date

# 1) ipfs_fetch.py with embedded template
ipfs_fetch = r'''#!/usr/bin/env python3
"""
ipfs_fetch.py
RAGE utility: fetch a ModelPack manifest + shards by CID with cache + verification.

Why this exists
- Universal access: any node can fetch the same model by CID.
- Safety: verify sha256 and refuse to load unknown/unverified shards.
- Scalability: shard-level caching (download once, reuse everywhere).

This tool is intentionally dependency-light:
- Uses stdlib + requests (optional).
- If `requests` is missing, it falls back to urllib.

Usage
  # Write a fresh template to MODEL_PACK.json (edit it, then publish to IPFS)
  python ipfs_fetch.py init-template --out MODEL_PACK.json

  # Fetch a manifest JSON from a gateway by CID and download shards into ./models/shards
  python ipfs_fetch.py fetch --manifest-cid <CID> --out-dir ./models

  # Fetch using a custom gateway
  python ipfs_fetch.py fetch --manifest-cid <CID> --gateway https://ipfs.io --out-dir ./models

Outputs
  out-dir/
    packs/<manifest-cid>.json
    shards/<filename-from-manifest>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
from urllib.parse import urljoin

# Optional requests for better UX
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


MODEL_PACK_TEMPLATE: Dict = {
    "format": "safetensors",
    "model_name": "ProductionTransformer-RAGE",
    "model_version": "1.1.0",
    "config": {
        "vocab_size": 32000,
        "d_model": 512,
        "num_layers": 8,
        "num_heads": 8,
        "num_kv_heads": 2,
        "dim_ff": 2048,
        "dropout": 0.0,
        "max_seq_len": 4096,
        "rope_theta": 10000.0,
        "tie_weights": True,
        "use_sdpa": True,
        "final_norm": True,
        "rmsnorm_eps": 1e-5
    },
    "shards": [
        {
            "filename": "model-00001-of-00002.safetensors",
            "cid": "bafyREPLACE_ME",
            "sha256": "sha256:REPLACE_ME"
        },
        {
            "filename": "model-00002-of-00002.safetensors",
            "cid": "bafyREPLACE_ME",
            "sha256": "sha256:REPLACE_ME"
        }
    ],
    "tokenizer": {
        "type": "sentencepiece|bpe|hf",
        "cid": "bafyOPTIONAL_TOKENIZER_CID"
    },
    "license": "REPLACE_ME",
    "provenance": {
        "build_ts": "2026-02-14T00:00:00Z",
        "git_commit": "REPLACE_ME",
        "trainer_wallet": "REPLACE_ME",
        "signature": "REPLACE_ME"
    }
}


def _mkdirp(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _http_get_bytes(url: str, timeout: int = 60) -> bytes:
    if requests is not None:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "RAGE-IPFS-Fetch/1.0"})
        r.raise_for_status()
        return r.content

    # urllib fallback
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "RAGE-IPFS-Fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_download_to(url: str, out_path: str, timeout: int = 300) -> None:
    # stream download if requests exists
    if requests is not None:
        with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "RAGE-IPFS-Fetch/1.0"}) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return

    # urllib fallback (no streaming progress)
    data = _http_get_bytes(url, timeout=timeout)
    with open(out_path, "wb") as f:
        f.write(data)


def _gateway_base(gateway: str) -> str:
    # Ensure it ends with /ipfs/
    g = gateway.rstrip("/")
    if not g.endswith("/ipfs"):
        g = g + "/ipfs"
    return g + "/"


def _ipfs_url(gateway: str, cid: str) -> str:
    return urljoin(_gateway_base(gateway), cid)


def init_template(out_path: str) -> None:
    _mkdirp(os.path.dirname(out_path) or ".")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(MODEL_PACK_TEMPLATE, f, indent=2)
    print(f"Wrote template: {out_path}")


def fetch_manifest(manifest_cid: str, gateway: str) -> Dict:
    url = _ipfs_url(gateway, manifest_cid)
    raw = _http_get_bytes(url, timeout=60)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse manifest JSON from {url}: {e}")


def fetch_shard(cid: str, filename: str, sha256_expected: Optional[str], gateway: str, shards_dir: str, force: bool = False) -> str:
    _mkdirp(shards_dir)
    out_path = os.path.join(shards_dir, filename)

    if os.path.exists(out_path) and not force:
        # Verify existing file before trusting it
        got = _sha256_file(out_path)
        if sha256_expected and got != sha256_expected:
            raise RuntimeError(f"Cached shard hash mismatch: {filename}\n expected {sha256_expected}\n got      {got}")
        return out_path

    # Download by CID (content addressing). Filename is purely local convenience.
    url = _ipfs_url(gateway, cid)
    tmp = out_path + ".partial"
    if os.path.exists(tmp):
        os.remove(tmp)

    print(f"Downloading shard: {filename}")
    _http_download_to(url, tmp, timeout=600)
    got = _sha256_file(tmp)

    if sha256_expected and got != sha256_expected:
        os.remove(tmp)
        raise RuntimeError(f"Downloaded shard hash mismatch: {filename}\n expected {sha256_expected}\n got      {got}")

    os.replace(tmp, out_path)
    return out_path


def fetch_modelpack(manifest_cid: str, out_dir: str, gateway: str, force: bool) -> str:
    packs_dir = os.path.join(out_dir, "packs")
    shards_dir = os.path.join(out_dir, "shards")
    _mkdirp(packs_dir)
    _mkdirp(shards_dir)

    manifest = fetch_manifest(manifest_cid, gateway=gateway)
    manifest_path = os.path.join(packs_dir, f"{manifest_cid}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    shards = manifest.get("shards", [])
    if not isinstance(shards, list) or not shards:
        raise RuntimeError("Manifest has no shards[] list.")

    for s in shards:
        cid = s.get("cid")
        fn = s.get("filename")
        sha = s.get("sha256")
        if not cid or not fn:
            raise RuntimeError(f"Invalid shard entry: {s}")
        fetch_shard(cid=cid, filename=fn, sha256_expected=sha, gateway=gateway, shards_dir=shards_dir, force=force)

    print(f"Fetched ModelPack manifest + shards into: {out_dir}")
    print(f"Manifest: {manifest_path}")
    return manifest_path


def main() -> int:
    p = argparse.ArgumentParser(description="RAGE IPFS ModelPack fetch utility")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-template", help="write a fresh ModelPack JSON template")
    p_init.add_argument("--out", default="MODEL_PACK.json", help="output path for template JSON")

    p_fetch = sub.add_parser("fetch", help="fetch manifest + shards by CID into a local directory")
    p_fetch.add_argument("--manifest-cid", required=True, help="CID of the ModelPack manifest JSON")
    p_fetch.add_argument("--gateway", default="https://ipfs.io", help="IPFS gateway base URL")
    p_fetch.add_argument("--out-dir", default="./models", help="directory to store packs/ and shards/")
    p_fetch.add_argument("--force", action="store_true", help="redownload shards even if cached")

    args = p.parse_args()

    try:
        if args.cmd == "init-template":
            init_template(args.out)
            return 0
        if args.cmd == "fetch":
            fetch_modelpack(args.manifest_cid, out_dir=args.out_dir, gateway=args.gateway, force=args.force)
            return 0
        raise RuntimeError("Unknown command")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''

# 2) generate.py: sampling + load modelpack (local) + optional tokenizer
generate_py = r'''#!/usr/bin/env python3
"""
generate.py
RAGE utility: run local generation with ProductionTransformerRAGE using KV cache + sampling.

This script is tokenizer-agnostic by default:
- You can pass `--prompt-ids '[1,2,3]'` (JSON list of ints)
- Or provide `--tokenizer <hf_name_or_path>` to use transformers.AutoTokenizer

Best practice for RAGE
- Stamp outputs with:
  - model_manifest_cid
  - shard filenames/cids
  - decoding params
  - timestamp
- Treat the run log as "memory" (append-only event)

Notes
- This is a lightweight reference. For high-throughput serving, prefer vLLM/llama.cpp/etc.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

# Local import: keep in same folder or adjust PYTHONPATH
from production_transformer_rage_v1.1.0 import (  # type: ignore
    ProductionTransformerRAGE,
    load_from_modelpack_local,
)


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """
    Apply top-k and/or nucleus (top-p) filtering to logits.
    logits: (B, V)
    returns filtered logits with -inf for removed tokens.
    """
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, top_k, dim=-1).values[..., -1].unsqueeze(-1)
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)

        # remove tokens with cumulative prob above top_p
        mask = cum > top_p
        # shift right to keep at least one token
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
        # scatter back
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    return logits


def apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    """
    Repetition penalty (simple version):
    - For tokens that already appeared in the sequence, divide positive logits by penalty and multiply negative logits by penalty.
    """
    if penalty == 1.0:
        return logits
    bsz = logits.size(0)
    for b in range(bsz):
        seen = torch.unique(generated[b])
        for tok in seen:
            tok = int(tok.item())
            val = logits[b, tok]
            logits[b, tok] = torch.where(val > 0, val / penalty, val * penalty)
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
    """
    Sampling decode with KV cache. prompt_ids: (B, T)
    """
    model.eval()
    device = prompt_ids.device
    out = prompt_ids
    past = None

    # Build cache with full prompt
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

        # decode only newest token, no causal needed
        logits, past = model(next_id, causal=False, use_cache=True, past_key_values=past)

    return out


def maybe_load_tokenizer(tokenizer_ref: Optional[str]):
    if not tokenizer_ref:
        return None
    try:
        from transformers import AutoTokenizer  # type: ignore
        return AutoTokenizer.from_pretrained(tokenizer_ref, use_fast=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer '{tokenizer_ref}': {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="RAGE local generation (ProductionTransformerRAGE)")
    ap.add_argument("--manifest", required=True, help="Path to ModelPack manifest JSON (local file)")
    ap.add_argument("--shard-dir", required=True, help="Directory containing shard files referenced by the manifest")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps (if available)")
    ap.add_argument("--tokenizer", default=None, help="Optional: HF tokenizer name/path for encode/decode")

    ap.add_argument("--prompt", default=None, help="Optional: prompt string (requires --tokenizer)")
    ap.add_argument("--prompt-ids", default=None, help="Optional: JSON list of token ids (no tokenizer required)")

    ap.add_argument("--max-new", type=int, default=64, help="Max new tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.8, help="0 for greedy, >0 for sampling")
    ap.add_argument("--top-k", type=int, default=40, help="Top-k sampling (0 disables)")
    ap.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling threshold")
    ap.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty (1.0 disables)")
    ap.add_argument("--eos", type=int, default=None, help="Optional eos token id (int)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (0 disables)")

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

    # Output
    if tok is not None:
        text = tok.decode(out_ids[0].tolist(), skip_special_tokens=True)
        print(text)
    else:
        print(json.dumps({"output_ids": out_ids[0].tolist()}, indent=2))

    # Optional run log
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
'''

# 3) PROMOTE_MODELPACK.md: promotion flow tying CID + signature into RAGE integrity tiers
promote_md = f"""# PROMOTE · ModelPack to IPFS (RAGE Best Practice)

**Last updated:** {date(2026,2,14).isoformat()}

This guide describes the recommended **industry-grade** workflow to publish model artifacts as **content-addressed ModelPacks** and promote them through RAGE integrity tiers.

## Why this matters
- **Universal access:** anyone can fetch the exact model by CID.
- **Integrity:** CIDs prove content immutability; hashes verify downloads.
- **Scalability:** shard-level caching allows global distribution.
- **Auditability:** every inference can be stamped with the manifest CID + signer.

---

## 1) Prepare artifacts (weights, config, tokenizer)

### 1.1 Use SafeTensors for weights
- Export weights in `safetensors` format.
- Prefer sharding so clients download in parallel and cache efficiently.

**Naming convention**
- `model-00001-of-000NN.safetensors`
- `model-00002-of-000NN.safetensors`
- ...

### 1.2 Compute hashes
Compute SHA-256 for each shard:

```bash
sha256sum model-*.safetensors
