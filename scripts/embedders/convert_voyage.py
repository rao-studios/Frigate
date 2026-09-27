#!/usr/bin/env python3
"""
WHAT: Build the MLX snapshot of voyage-4-nano that Frigate's embedder downloads —
      `rao-studios/voyage-4-nano-mlx-<precision>` — from the pinned upstream revision.
IN:   `voyageai/voyage-4-nano` @ UPSTREAM_REVISION (fetched with huggingface_hub, or a
      local snapshot via --source), a precision (bf16 | 8bit), an output directory.
OUT:  A folder ready for `hf upload`: model.safetensors, config.json (auto_map removed;
      `quantization` added for 8bit), the tokenizer files, the prompts
      (config_sentence_transformers.json) and pooling config, LICENSE.txt verbatim,
      NOTICE.txt (upstream + Rao Studios' modification statement, Apache-2.0 §4(b)),
      and a README model card. Parity numbers go in with --parity <json>.
PIN:  Plain `mlx`, not `mlx_lm.convert`: that refuses the bare `linear` head and can
      save float16, which overflows this model. bf16 copies the upstream weights
      byte for byte. 8bit quantizes every projection and the token embedding
      (group 64, affine) and leaves the norms and the `linear` head in bfloat16 —
      one global quantization block, which is all Frigate's loader reads.

  python3 convert_voyage.py --precision 8bit --out build/voyage-4-nano-mlx-8bit
  python3 convert_voyage.py --precision bf16 --out build/voyage-4-nano-mlx-bf16 --parity parity.json
"""
import argparse, hashlib, json, pathlib, shutil, sys

UPSTREAM = "voyageai/voyage-4-nano"
UPSTREAM_REVISION = "67fabc9bef010dabc5f6024aa1b1b6b93410426f"
GROUP_SIZE = 64

COPIED = [
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "config_sentence_transformers.json", "1_Pooling/config.json", "LICENSE.txt",
]
QUANTIZED_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight", "embed_tokens.weight",
)

MODIFICATION_NOTICE = """
============================================================
= Modifications by Rao Studios                              =
============================================================
This repository redistributes voyage-4-nano converted for Apple's MLX framework.
Source: https://huggingface.co/{upstream} at revision {revision}.
Changes: {changes}
The conversion script is `scripts/embedders/convert_voyage.py` in
https://github.com/rao-studios/Frigate (sha256 {script_sha}).
This is an unofficial conversion, not affiliated with or endorsed by MongoDB, Inc.
or Voyage AI.
"""


def fetch_source(source):
    if source:
        return pathlib.Path(source)
    from huggingface_hub import snapshot_download
    return pathlib.Path(snapshot_download(
        UPSTREAM, revision=UPSTREAM_REVISION,
        allow_patterns=["*.safetensors", "*.json", "*.txt"]))


def convert_weights(src, out, precision):
    if precision == "bf16":
        shutil.copyfile(resolve(src, "model.safetensors"), out / "model.safetensors")
        return "none — the weights are the upstream bfloat16 tensors, byte for byte"
    import mlx.core as mx
    weights = mx.load(str(resolve(src, "model.safetensors")))
    converted = {}
    for key, value in weights.items():
        if value.dtype == mx.float16:
            value = value.astype(mx.bfloat16)
        if value.ndim == 2 and key.endswith(QUANTIZED_SUFFIXES):
            wq, scales, biases = mx.quantize(value, group_size=GROUP_SIZE, bits=8)
            stem = key[: -len("weight")]
            converted[stem + "weight"] = wq
            converted[stem + "scales"] = scales
            converted[stem + "biases"] = biases
        else:
            converted[key] = value
    mx.save_safetensors(str(out / "model.safetensors"), converted, metadata={"format": "mlx"})
    return ("projections and the token embedding quantized to 8-bit (affine, group 64); "
            "norms and the linear head kept in bfloat16; config `auto_map` removed")


def card(precision, parity, script_sha, size_mb):
    quant = "" if precision == "bf16" else "\nbase_model_relation: quantized"
    table = ""
    if parity:
        rows = "\n".join(
            f"| {name} | {v['reference']:.4f} | {v['this']:.4f} |" for name, v in parity["ndcg10"].items())
        table = f"""
## Parity

Measured with Frigate's `FrigateEmbedder` (this snapshot, 1024 dims, 512 tokens) against
sentence-transformers running the upstream model in bfloat16. nDCG@10:

| Dataset | Upstream (sentence-transformers) | This snapshot (Frigate/MLX) |
|---|---|---|
{rows}

Mean cosine to the upstream vectors: {parity['cos_mean']:.5f} (minimum {parity['cos_min']:.5f}); no
non-finite values.
"""
    return f"""---
license: apache-2.0
library_name: mlx
base_model: voyageai/voyage-4-nano{quant}
pipeline_tag: feature-extraction
language:
- multilingual
tags:
- mlx
- embeddings
- sentence-transformers
---

# voyage-4-nano (MLX, {precision})

An MLX conversion of [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano)
at revision `{UPSTREAM_REVISION}`, packaged for on-device embedding in
[Ambient](https://ambient.rao.nyc) through [Frigate](https://github.com/rao-studios/Frigate).
Unofficial: not affiliated with or endorsed by MongoDB, Inc. or Voyage AI.

- Weights: {precision} (`model.safetensors`, {size_mb:.0f} MB).
- Architecture: Qwen3, 12 layers, bidirectional attention, a 1024→2048 linear head, mean
  pooling (prompt included), L2 normalisation. Matryoshka: the first 1024, 512 or 256
  dimensions, renormalised, are embeddings too.
- Prompts (`config_sentence_transformers.json`):
  - query: `Represent the query for retrieving supporting documents: `
  - document: `Represent the document for retrieval: `
- Shares an embedding space with Voyage's hosted voyage-4, voyage-4-lite and voyage-4-large.
- Run it in bfloat16 or quantized; float16 overflows on ordinary text.

## Use with Frigate

```swift
let embedder = FrigateEmbedder(profile: .voyage4Nano)
let queries = try await embedder.embed(["what is a lichen?"], role: .query)
let passages = try await embedder.embed(["Lichens are a symbiosis of a fungus and an alga."], role: .document)
```
{table}
## Licence

Apache-2.0, as upstream. `LICENSE.txt` is the upstream licence and `NOTICE.txt` carries the
upstream notice (MongoDB, Inc.; built on Qwen3 by Alibaba Cloud) followed by Rao Studios'
statement of what was changed. Conversion script: `scripts/embedders/convert_voyage.py` in
rao-studios/Frigate, sha256 `{script_sha}`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["bf16", "8bit"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", help="a local upstream snapshot instead of downloading")
    ap.add_argument("--parity", help="JSON with ndcg10 {dataset: {reference, this}}, cos_mean, cos_min")
    args = ap.parse_args()

    src = fetch_source(args.source)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    script_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()

    for name in COPIED:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolve(src, name), target)

    changes = convert_weights(src, out, args.precision)

    config = json.loads(resolve(src, "config.json").read_text())
    config.pop("auto_map", None)
    if args.precision == "8bit":
        config["quantization"] = {"group_size": GROUP_SIZE, "bits": 8}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    notice = resolve(src, "NOTICE.txt").read_text().rstrip() + "\n" + MODIFICATION_NOTICE.format(
        upstream=UPSTREAM, revision=UPSTREAM_REVISION, changes=changes, script_sha=script_sha)
    (out / "NOTICE.txt").write_text(notice)

    parity = json.loads(pathlib.Path(args.parity).read_text()) if args.parity else None
    size_mb = (out / "model.safetensors").stat().st_size / 1e6
    (out / "README.md").write_text(card(args.precision, parity, script_sha, size_mb))
    print(f"wrote {out} ({size_mb:.0f} MB weights, {args.precision})")


def resolve(src, name):
    """The upstream file, from the snapshot or, for files a sentence-transformers cache
    never fetched (LICENSE/NOTICE), straight from the pinned revision."""
    local = src / name
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download
    return pathlib.Path(hf_hub_download(UPSTREAM, name, revision=UPSTREAM_REVISION))


if __name__ == "__main__":
    sys.exit(main())
