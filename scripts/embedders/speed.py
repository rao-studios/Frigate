# Throughput, memory and numeric health of one model at one precision, alone on
# the GPU: 512 SciFact documents (≤1800 chars, 512-token cap) and 300 queries.
#   .venv/bin/python speed.py <model-id> <float32|bfloat16|float16> [remote]
import json, sys, time, pathlib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

D = pathlib.Path(__file__).parent / 'data'
mid, dtype = sys.argv[1], getattr(torch, sys.argv[2])
remote = len(sys.argv) > 3
docs = [json.loads(l)['text'] for l in open(D / 'scifact_corpus.jsonl')][:512]
queries = [json.loads(l)['text'] for l in open(D / 'scifact_queries.jsonl')]

t0 = time.time()
model = SentenceTransformer(mid, trust_remote_code=remote, device='mps', truncate_dim=1024,
                            model_kwargs={'torch_dtype': dtype})
model.max_seq_length = 512
load_s = time.time() - t0
params = sum(p.numel() for p in model.parameters())

model.encode_document(docs[:32], batch_size=16)          # warm-up: kernels, allocator
torch.mps.synchronize()

def timed(fn, texts):
    torch.mps.synchronize(); t = time.time()
    v = np.asarray(fn(texts, batch_size=16, convert_to_numpy=True), dtype=np.float32)
    torch.mps.synchronize()
    return v, len(texts) / (time.time() - t)

dv, dps = timed(model.encode_document, docs)
qv, qps = timed(model.encode_query, queries)
nan = int(np.isnan(dv).any(1).sum() + np.isnan(qv).any(1).sum())
mem = torch.mps.driver_allocated_memory() / 2**30
print(json.dumps({'model': mid.split('/')[-1], 'dtype': sys.argv[2], 'params_M': round(params / 1e6),
                  'load_s': round(load_s, 1), 'docs_per_s': round(dps, 1), 'queries_per_s': round(qps, 1),
                  'gpu_mem_GiB': round(mem, 2), 'nan_rows': nan}))
