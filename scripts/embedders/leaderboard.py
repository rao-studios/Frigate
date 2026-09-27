# nDCG@10 for every model that has vectors for a set, against mistral-embed.
import json, pathlib
import numpy as np
import evaluate as ev

SETS = ['scifact', 'nfcorpus', 'germanquad', 'alloprof']
ROWS = [  # tag, query-file suffix, label, license, languages
    ('mistral', '', 'mistral-embed (hosted, today)', 'API', 'multi'),
    ('voyage4nano', '', 'voyage-4-nano @1024 (bf16)', 'Apache-2.0', 'multi'),
    ('pplx06', '', 'pplx-embed-v1-0.6B (fp32)', 'MIT', 'multi'),
    ('pplx06bf', '', 'pplx-embed-v1-0.6B (bf16)', 'MIT', 'multi'),
    ('arcticl2', '', 'snowflake-arctic-embed-l-v2.0', 'Apache-2.0', 'multi'),
    ('me5inst', '', 'multilingual-e5-large-instruct', 'MIT', 'multi'),
    ('bgem3', '', 'bge-m3', 'MIT', 'multi'),
    ('mxbai', '', 'mxbai-embed-large-v1', 'Apache-2.0', 'English'),
    ('qwen06st', '', 'Qwen3-Embedding-0.6B (+instr, reference)', 'Apache-2.0', 'multi'),
    ('qwen06', '_ins', 'Qwen3-Embedding-0.6B (+instr, Frigate)', 'Apache-2.0', 'multi'),
    ('qwen06', '', 'Qwen3-Embedding-0.6B (as Thread runs it)', 'Apache-2.0', 'multi'),
    ('qwen4b', '_ins', 'Qwen3-Embedding-4B @1024 (+instr)', 'Apache-2.0', 'multi'),
]


def score(tag, qsuf, ds):
    qf, cf = ev.D / f'{ds}_queries{qsuf}.{tag}.f32', ev.D / f'{ds}_corpus.{tag}.f32'
    if not (qf.exists() and cf.exists()):
        return None
    qrels = json.load(open(ev.D / f'{ds}_qrels.json'))
    q = ev.unit(ev.load(f'{ds}_queries{qsuf}', tag)[:, :1024])
    d = ev.unit(ev.load(f'{ds}_corpus', tag)[:, :1024])
    return ev.ndcg10(q, d, ev.ids(f'{ds}_queries'), ev.ids(f'{ds}_corpus'), qrels)[0]


base = {ds: score('mistral', '', ds) for ds in SETS}
print(f"{'model':42s}" + ''.join(f'{s:>12s}' for s in SETS) + f"{'avg':>8s}{'vs mistral':>12s}")
for tag, qsuf, label, lic, langs in ROWS:
    vals = [score(tag, qsuf, ds) for ds in SETS]
    have = [(v, base[s]) for v, s in zip(vals, SETS) if v is not None]
    avg = np.mean([v for v, _ in have]) if have else None
    rel = np.mean([v / b for v, b in have]) if have else None
    cells = ''.join(f'{v:12.3f}' if v is not None else f"{'—':>12s}" for v in vals)
    tail = (f'{avg:8.3f}{rel * 100:11.0f}%' if have and len(have) == len(SETS) else f"{'':8s}{(f'{rel*100:.0f}%*' if have else ''):>12s}")
    print(f'{label:42s}{cells}{tail}   {lic}, {langs}')
print('* partial: averaged over the sets it has')
