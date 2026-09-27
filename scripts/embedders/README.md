# Embedder checks and the voyage-4-nano snapshot

`FrigateEmbedder`'s default model is `rao-studios/voyage-4-nano-mlx-8bit`: an MLX
conversion of [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano)
(Apache-2.0) at revision `67fabc9bef010dabc5f6024aa1b1b6b93410426f`. This folder builds
that snapshot and holds the checks a build must pass before it ships.

## Why this model

These are nDCG@10 averages over SciFact, NFCorpus, GermanQuAD and Alloprof, with every
model at 1024 dimensions and 512 tokens, measured 2026-09-27 against mistral-embed:

| Model | vs mistral-embed |
|---|---|
| voyage-4-nano (bf16) | 96% |
| pplx-embed-v1-0.6B | 96% |
| Qwen3-Embedding-0.6B, reference implementation | 94% |
| Qwen3-Embedding-0.6B, Frigate before the truncation fix | 77% |

voyage-4-nano is also the fastest here: ~47 docs/s bf16 and ~43 docs/s 8-bit on an M4 Max,
346M parameters. It shares an embedding space with Voyage's hosted voyage-4 models.

## Steps

```bash
cd scripts/embedders
python3 prepare_data.py                       # benchmark sets → data/ (gitignored)
python3 -m venv .venv && .venv/bin/pip install torch "transformers<5" sentence-transformers numpy einops

# 1. Reference vectors: the upstream model under sentence-transformers, bf16.
.venv/bin/python st_embed.py voyage4nano scifact_queries nfcorpus_queries germanquad_queries alloprof_queries \
    scifact_corpus nfcorpus_corpus germanquad_corpus alloprof_corpus

# 2. Build a snapshot (bf16 copies the weights; 8bit quantizes projections + embedding).
python3 convert_voyage.py --precision 8bit --out build/voyage-4-nano-mlx-8bit

# 3. The same texts through FrigateEmbedder (the path Thread takes).
(cd embedprobe && swift build -c release && bash ../../build-metallib.sh release --package .)
P=embedprobe/.build/release/embedprobe
for ds in scifact nfcorpus germanquad alloprof; do
  $P build/voyage-4-nano-mlx-8bit query    data/${ds}_queries.jsonl data/${ds}_queries.f8bit.f32 --dim 1024
  $P build/voyage-4-nano-mlx-8bit document data/${ds}_corpus.jsonl  data/${ds}_corpus.f8bit.f32  --dim 1024
done

# 4. Parity. Exit status 0 means it passes.
python3 parity.py voyage4nano f8bit --quantized --json build/parity-8bit.json

# 5. Rebuild with the parity table on the card, then publish (public repo, rao-studios).
python3 convert_voyage.py --precision 8bit --out build/voyage-4-nano-mlx-8bit --parity build/parity-8bit.json
hf upload rao-studios/voyage-4-nano-mlx-8bit build/voyage-4-nano-mlx-8bit .
```

After publishing, set `FrigateEmbedder.Profile.voyage4NanoRevision` to the repo's commit
sha. That pin is the one place a new conversion ships from.

## Targets

| | bf16 | quantized |
|---|---|---|
| mean cosine to the reference | ≥ 0.999 | ≥ 0.995 |
| minimum cosine | ≥ 0.995 | ≥ 0.98 |
| nDCG@10 drop, averaged over the four sets | ≤ 0.005 | ≤ 0.005 |
| nDCG@10 drop, worst set | ≤ 0.01 | ≤ 0.01 |
| non-finite rows | 0 | 0 |

Ship the smallest precision that passes. On 2026-09-27 the 8-bit build scored mean cosine
0.99956 (minimum 0.99810), with a worst nDCG drop of 0.0009.

## Other scripts

- `leaderboard.py`: nDCG@10 for every model with vectors in `data/`.
- `speed.py`: docs/s, queries/s and memory for a sentence-transformers model at a given dtype.
- `mistral_embed.py`: mistral-embed baselines. Needs a key in `~/.rao/keys/providers.json`.
- `evaluate.py`: the shared loaders and nDCG, plus the (failed) experiment of mapping another
  model into mistral-embed's space.

## Notes

- **float16:** voyage-4-nano and pplx-embed both overflow float16 on ordinary text and return
  NaN rows. Run bf16 or quantized. `FrigateEmbedder` throws on non-finite output.
- **Licence:** the snapshot carries `LICENSE.txt` verbatim, plus `NOTICE.txt` (the upstream
  notice followed by Rao Studios' modification statement, Apache-2.0 §4(b)).
