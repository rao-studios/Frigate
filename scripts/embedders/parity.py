#!/usr/bin/env python3
"""
WHAT: Does a Frigate build of a model match the reference? Cosine per text and nDCG@10
      per set between two vector tags in data/ (e.g. the sentence-transformers reference
      `voyage4nano` and an embedprobe run `f8bit`), and the verdict against the targets.
OUT:  A table on stdout; with --json, the parity block convert_voyage.py puts on the card.
PIN:  Targets — bf16: mean cos ≥ 0.999, min ≥ 0.995; quantized: mean ≥ 0.995, min ≥ 0.98;
      both: average nDCG drop ≤ 0.005, worst set ≤ 0.01, no non-finite rows.

  python3 parity.py voyage4nano f8bit --quantized --json build/parity-8bit.json
"""
import argparse, json
import numpy as np
import evaluate as ev

SETS = ["scifact", "nfcorpus", "germanquad", "alloprof"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--quantized", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()

    cosines, nonfinite, ndcg = [], 0, {}
    for ds in SETS:
        qrels = json.load(open(ev.D / f"{ds}_qrels.json"))
        qids, dids = ev.ids(f"{ds}_queries"), ev.ids(f"{ds}_corpus")
        scores = {}
        for tag in (args.reference, args.candidate):
            q, d = ev.load(f"{ds}_queries", tag), ev.load(f"{ds}_corpus", tag)
            if tag == args.candidate:
                nonfinite += int((~np.isfinite(q)).any(1).sum() + (~np.isfinite(d)).any(1).sum())
            scores[tag] = ev.ndcg10(ev.unit(q), ev.unit(d), qids, dids, qrels)[0]
        for kind in ("queries", "corpus"):
            a = ev.unit(ev.load(f"{ds}_{kind}", args.reference))
            b = ev.unit(ev.load(f"{ds}_{kind}", args.candidate))
            cosines.append(np.sum(a * b, axis=1))
        ndcg[ds] = {"reference": scores[args.reference], "this": scores[args.candidate]}

    cos = np.concatenate(cosines)
    drops = [v["reference"] - v["this"] for v in ndcg.values()]
    mean_floor, min_floor = (0.995, 0.98) if args.quantized else (0.999, 0.995)
    checks = {
        f"mean cos ≥ {mean_floor}": cos.mean() >= mean_floor,
        f"min cos ≥ {min_floor}": cos.min() >= min_floor,
        "average nDCG drop ≤ 0.005": np.mean(drops) <= 0.005,
        "worst set drop ≤ 0.01": max(drops) <= 0.01,
        "no non-finite rows": nonfinite == 0,
    }
    print(f"{'set':12s}{'reference':>11s}{'candidate':>11s}{'drop':>8s}")
    for ds, v in ndcg.items():
        print(f"{ds:12s}{v['reference']:11.4f}{v['this']:11.4f}{v['reference'] - v['this']:8.4f}")
    print(f"cosine: mean {cos.mean():.5f}, min {cos.min():.5f}, p1 {np.percentile(cos, 1):.5f}; non-finite rows {nonfinite}")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    verdict = all(checks.values())
    print("verdict:", "PASS" if verdict else "FAIL")
    if args.json:
        json.dump({"ndcg10": ndcg, "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
                   "pass": bool(verdict)}, open(args.json, "w"), indent=2)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
