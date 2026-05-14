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
