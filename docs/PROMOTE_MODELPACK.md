# PROMOTE · ModelPack to IPFS (RAGE Best Practice)

**Last updated:** 2026-02-14

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
```

Record results as `sha256:<hex>`.

---

## 2) Add shards to IPFS (get CIDs)

```bash
ipfs add --pin=false model-00001-of-000NN.safetensors
ipfs add --pin=false model-00002-of-000NN.safetensors
```

Capture each CID and map it to the shard filename.

> Best practice: pin later in a controlled step. `--pin=false` keeps publishing separate from persistence policy.

---

## 3) Create the ModelPack manifest JSON

Start with a template:

```bash
python ipfs_fetch.py init-template --out MODEL_PACK.json
```

Then edit:
- `config` (must match the model code expectations)
- `shards[]` entries: `{filename, cid, sha256}`
- optional `tokenizer` entry
- `license` and `provenance`

---

## 4) Publish the manifest to IPFS (manifest CID = model version)

```bash
ipfs add --pin=false MODEL_PACK.json
```

The returned CID is your **model version ID**.

**Recommendation**
- Treat this manifest as a RAGE **Capsule**:
  - L2 = manifest CID exists
  - L3 = maintainer signature added
  - L4 = validator quorum
  - L5 = promoted / anchored (optional)

---

## 5) Sign the manifest (authorship / attestation)

### 5.1 Canonical signing payload
Sign the hash of a canonical payload:
- `manifest_cid`
- `bundle_root` (optional)
- `timestamp`
- `origin`
- `nonce`

Example payload (JSON):
```json
{
  "v": 1,
  "app": "RAGE",
  "origin": "rage.pythai.net",
  "action": "publish_modelpack",
  "manifest_cid": "bafy...",
  "nonce": "random-unique",
  "ts": "2026-02-14T00:00:00Z"
}
```

Store the signature in:
- the manifest `provenance.signature`, or
- a parallel `SIGNATURE.json` pinned to IPFS and referenced from the manifest.

### 5.2 RAGE receipts
When a maintainer signs a manifest, record a **Receipt**:
- wallet address
- signature
- payload hash
- manifest CID
- timestamp + nonce

This enables RAGE to display “Verified Model” badges and to stamp inference runs with verifiable provenance.

---

## 6) Publish “latest” pointers (because CIDs are immutable)

CIDs never change. To publish “latest”:
- update a pointer layer:
  - DNSLink
  - IPNS
  - a signed `latest.json` (itself pinned; pointer updated)

**Rule**
- Move the pointer, never mutate the CID.

---

## 7) Pinning strategy (durability policy)

Choose at least one:
- pin on an org IPFS node
- pin on multiple community nodes
- use a pinning provider
- publish CAR files for bulk replication

**Best practice**
- Pin **manifest** and **all shards** together.

---

## 8) Verification (clients must enforce)

Clients should:
1) fetch manifest by CID
2) download shards by CID
3) verify `sha256` matches manifest
4) load `safetensors` only after verification

The provided `ipfs_fetch.py` does exactly this (cache-first + sha256 verification).

---

## 9) Operational workflow inside RAGE

### 9.1 Treat model publishes as first-class memory events
Log events like:
- `MODEL_SHARD_PUBLISHED`
- `MODELPACK_PUBLISHED`
- `MODELPACK_SIGNED`
- `MODELPACK_VALIDATED`
- `MODELPACK_PROMOTED`

Because **logs are memory**, this gives you:
- traceability from “who published which model”
- ability to roll back by switching pointers
- reproducible inference (always re-fetch by manifest CID)

### 9.2 Stamp inference outputs
Every inference log should include:
- `modelpack_manifest_cid`
- `decode_params`
- `timestamp`
- optional `signature` (for high-assurance runs)

---

## 10) Minimal folder layout (recommended)

```
neuralnet/
  tools/
    ipfs_fetch.py
    generate.py
  models/
    packs/
      <manifest_cid>.json
    shards/
      model-00001-of-000NN.safetensors
```

---

## 11) Next upgrades
- Add CAR file packaging and resumable gateway fetches
- Add signature verification module (EVM/Solana adapters)
- Add validator quorum and reputation weighting in RAGE
- Add quantized ModelPacks (int8/int4) for universal CPU access
