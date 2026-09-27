# Can an on-device embedder live in mistral-embed's space?
# Fit adapters Local -> Mistral on NQ (train.*), then score retrieval on SciFact
# and NFCorpus in every mix a Thread table can hold.
#   python3 evaluate.py <local-tag> [<local-tag> ...]
import json, sys, pathlib
import numpy as np

D = pathlib.Path(__file__).parent / 'data'
rng = np.random.default_rng(7)


def load(name, tag):
    a = np.fromfile(D / f'{name}.{tag}.f32', dtype=np.float32)
    n = sum(1 for _ in open(D / f'{name}.jsonl'))
    return a.reshape(n, -1)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)


def ids(name):
    return [json.loads(l).get('id') for l in open(D / f'{name}.jsonl')]


# ---------- adapters ----------

def fit_procrustes(X, Y):
    """Orthogonal map after centring both sides (needs equal dims)."""
    mx, my = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((X - mx).T @ (Y - my), full_matrices=False)
    W = U @ Vt
    return lambda Z: unit((Z - mx) @ W + my)


def fit_ridge(X, Y, lam):
    """Affine least squares, then back onto the sphere."""
    Xa = np.hstack([X, np.ones((len(X), 1), X.dtype)])
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1], dtype=np.float64)
    W = np.linalg.solve(A, Xa.T @ Y).astype(np.float32)
    return lambda Z: unit(np.hstack([Z, np.ones((len(Z), 1), Z.dtype)]) @ W)


def pick_ridge(X, Y):
    n = len(X); idx = rng.permutation(n); cut = int(n * 0.9)
    tr, va = idx[:cut], idx[cut:]
    best = None
    for lam in (0.01, 0.1, 1, 3, 10, 30, 100):
        f = fit_ridge(X[tr], Y[tr], lam)
        cos = float(np.mean(np.sum(f(X[va]) * Y[va], axis=1)))
        if best is None or cos > best[1]: best = (lam, cos)
    return best[0], fit_ridge(X, Y, best[0])


# ---------- retrieval ----------

def ndcg10(Qv, Dv, qids, dids, qrels):
    S = Qv @ Dv.T
    top = np.argsort(-S, axis=1)[:, :10]
    scores = []
    for i, qid in enumerate(qids):
        rel = qrels.get(qid, {})
        if not rel: continue
        dcg = sum(rel.get(dids[j], 0) / np.log2(r + 2) for r, j in enumerate(top[i]))
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
        scores.append(dcg / idcg)
    return float(np.mean(scores)), top


def run(tag):
    Mtr, Ltr = unit(load('train', 'mistral')), unit(load('train', tag))
    lam, ridge = pick_ridge(Ltr, Mtr)
    adapters = {'ridge(λ=%g)' % lam: ridge}
    if Ltr.shape[1] == Mtr.shape[1]:
        adapters['procrustes'] = fit_procrustes(Ltr, Mtr)
    print(f'\n=== local model: {tag} (dim {Ltr.shape[1]}) — adapters fit on {len(Ltr)} NQ texts ===')
    for ds in ('scifact', 'nfcorpus'):
        qrels = json.load(open(D / f'{ds}_qrels.json'))
        qids, dids = ids(f'{ds}_queries'), ids(f'{ds}_corpus')
        Mq, Md = unit(load(f'{ds}_queries', 'mistral')), unit(load(f'{ds}_corpus', 'mistral'))
        Lq, Ld = unit(load(f'{ds}_queries', tag)), unit(load(f'{ds}_corpus', tag))
        base_m, _ = ndcg10(Mq, Md, qids, dids, qrels)
        base_l, _ = ndcg10(Lq, Ld, qids, dids, qrels)
        print(f'\n[{ds}] {len(qids)} queries × {len(dids)} docs — nDCG@10')
        print(f'  mistral → mistral                 {base_m:.3f}   (Plus today)')
        print(f'  local   → local                   {base_l:.3f}   (on-device, own space)')
        half = rng.random(len(dids)) < 0.5
        for name, f in adapters.items():
            Aq, Ad = f(Lq), f(Ld)
            fid = float(np.mean(np.sum(f(np.vstack([Lq, Ld])) * np.vstack([Mq, Md]), axis=1)))
            lapse, _ = ndcg10(Aq, Md, qids, dids, qrels)
            upgrade, _ = ndcg10(Mq, Ad, qids, dids, qrels)
            aa, _ = ndcg10(Aq, Ad, qids, dids, qrels)
            mixed = np.where(half[:, None], Md, Ad)
            mix_m, top_m = ndcg10(Mq, mixed, qids, dids, qrels)
            mix_a, top_a = ndcg10(Aq, mixed, qids, dids, qrels)
            share_m = float(half[top_m].mean()); share_a = float(half[top_a].mean())
            print(f'  -- adapter {name}: mean cos(adapted, mistral) on eval texts = {fid:.3f}')
            print(f'  adapted → adapted                 {aa:.3f}')
            print(f'  adapted query → mistral docs      {lapse:.3f}   (Plus lapsed: on-device over Plus-filed)')
            print(f'  mistral query → adapted docs      {upgrade:.3f}   (upgraded: Plus over on-device-filed)')
            print(f'  half/half table, mistral query    {mix_m:.3f}   top-10 native share {share_m:.2f} (0.50 = unbiased)')
            print(f'  half/half table, adapted query    {mix_a:.3f}   top-10 native share {share_a:.2f}')


if __name__ == '__main__':
    for tag in sys.argv[1:]:
        run(tag)
