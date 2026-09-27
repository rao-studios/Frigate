# Embed each JSONL file with mistral-embed; write <name>.mistral.f32 in input order.
import json, sys, time, os, pathlib, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np

D = pathlib.Path(sys.argv[1])
KEY = json.load(open(os.path.expanduser('~/.rao/keys/providers.json')))['MISTRAL_API_KEY']['value']
URL = 'https://api.mistral.ai/v1/embeddings'

def post(texts):
    body = json.dumps({'model': 'mistral-embed', 'input': texts}).encode()
    for attempt in range(8):
        req = urllib.request.Request(URL, data=body, headers={
            'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            data = sorted(out['data'], key=lambda d: d['index'])
            return [d['embedding'] for d in data], out['usage']['prompt_tokens']
        except urllib.error.HTTPError as e:
            msg = e.read()[:200]
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt); continue
            raise RuntimeError(f'HTTP {e.code}: {msg!r}')
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError('gave up after retries')

def batches(texts, max_n=32, max_chars=36000):
    cur, size, start = [], 0, 0
    for i, t in enumerate(texts):
        if cur and (len(cur) >= max_n or size + len(t) > max_chars):
            yield start, cur; start, cur, size = i, [], 0
        cur.append(t); size += len(t)
    if cur: yield start, cur

total_tokens = 0
for name in sys.argv[2:]:
    out_path = D / f'{name}.mistral.f32'
    if out_path.exists(): print(name, 'exists, skipped'); continue
    texts = [json.loads(l)['text'] for l in open(D / f'{name}.jsonl')]
    vecs = [None] * len(texts)
    jobs = list(batches(texts))
    t0 = time.time()
    with ThreadPoolExecutor(3) as pool:
        for (start, chunk), (emb, tok) in zip(jobs, pool.map(lambda j: post(j[1]), jobs)):
            vecs[start:start + len(chunk)] = emb
            total_tokens += tok
    arr = np.asarray(vecs, dtype=np.float32)
    arr.tofile(out_path)
    print(f'{name}: {arr.shape} in {time.time()-t0:.0f}s; tokens so far {total_tokens}', flush=True)
print('done, total tokens', total_tokens)
