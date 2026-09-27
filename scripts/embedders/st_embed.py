# Embed the benchmark sets with a sentence-transformers model, each with its own
# documented prompts and pooling, truncated to 512 tokens (Frigate's cap) and
# cut to 1024 dims where the model is larger. Writes data/<set>.<tag>.f32.
#   .venv/bin/python st_embed.py <tag> <set> [<set> ...]
import json, sys, time, pathlib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

D = pathlib.Path(__file__).parent / 'data'
E5 = 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: '

MODELS = {
    'voyage4nano': dict(id='voyageai/voyage-4-nano', remote=True, dim=1024, dtype='bfloat16'),
    'pplx06bf':    dict(id='perplexity-ai/pplx-embed-v1-0.6b', remote=True, dim=1024, dtype='bfloat16'),
    'pplx06':      dict(id='perplexity-ai/pplx-embed-v1-0.6b', remote=True, dim=1024, fp32=True),
    'arcticl2':    dict(id='Snowflake/snowflake-arctic-embed-l-v2.0', remote=False, dim=1024),
    'bgem3':       dict(id='BAAI/bge-m3', remote=False, dim=1024),
    'me5inst':     dict(id='intfloat/multilingual-e5-large-instruct', remote=False, dim=1024, query_prompt=E5),
    'qwen06st':    dict(id='Qwen/Qwen3-Embedding-0.6B', remote=False, dim=1024,
                        query_prompt='Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'),
    'mxbai':       dict(id='mixedbread-ai/mxbai-embed-large-v1', remote=False, dim=1024),
}

tag, sets = sys.argv[1], sys.argv[2:]
spec = MODELS[tag]
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = SentenceTransformer(spec['id'], trust_remote_code=spec['remote'], device=device,
                            truncate_dim=spec['dim'], model_kwargs={'torch_dtype': torch.float32 if spec.get('fp32') else getattr(torch, spec.get('dtype', 'float16'))})
model.max_seq_length = 512
print(f'{tag}: {spec["id"]} on {device}, prompts={getattr(model, "prompts", {})}', flush=True)

for name in sets:
    out = D / f'{name}.{tag}.f32'
    if out.exists():
        continue
    texts = [json.loads(l)['text'] for l in open(D / f'{name}.jsonl')]
    is_query = '_queries' in name
    t0 = time.time()
    kwargs = dict(batch_size=16, convert_to_numpy=True, show_progress_bar=False)
    if is_query and 'query_prompt' in spec:
        vecs = model.encode(texts, prompt=spec['query_prompt'], **kwargs)
    elif is_query:
        vecs = model.encode_query(texts, **kwargs)
    else:
        vecs = model.encode_document(texts, **kwargs)
    vecs = np.asarray(vecs, dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(1e-9)
    assert vecs.shape == (len(texts), spec['dim']), vecs.shape
    vecs.tofile(out)
    print(f'  {name}: {vecs.shape} in {time.time() - t0:.0f}s ({len(texts) / (time.time() - t0):.1f}/s)', flush=True)
