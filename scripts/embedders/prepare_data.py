#!/usr/bin/env python3
"""
WHAT: Fetch the retrieval sets the embedder checks run on into data/: SciFact and
      NFCorpus (English science), GermanQuAD (German) and Alloprof (French), plus
      10k Natural Questions texts for fitting experiments.
OUT:  data/<set>_corpus.jsonl, data/<set>_queries.jsonl ({"id", "text"}), data/<set>_qrels.json.
PIN:  Texts are cut to 1800 characters — about the 512-token budget FrigateEmbedder
      embeds — so every model sees the same words.

  python3 prepare_data.py
"""
import json, pathlib, random
from datasets import load_dataset

D = pathlib.Path(__file__).parent / "data"
MAXC = 1800
SETS = {
    "scifact": "mteb/scifact",
    "nfcorpus": "mteb/nfcorpus",
    "germanquad": "mteb/GermanQuAD-Retrieval",
    "alloprof": "mteb/AlloprofRetrieval",
}


def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def first(ds):
    return ds[list(ds.keys())[0]]


def qrels_for(repo):
    attempts = (
        lambda: load_dataset(repo, "default"),
        lambda: load_dataset(repo + "-qrels"),
        lambda: load_dataset(repo, data_files={"test": "qrels/*.parquet"}),
    )
    last = None
    for attempt in attempts:
        try:
            d = attempt()
            return d["test"] if "test" in d else first(d)
        except Exception as e:  # the MTEB repos lay qrels out three ways
            last = e
    raise last


def text_of(row):
    title = (row.get("title") or "").strip()
    body = (row.get("text") or "").strip()
    return ((title + ". " + body) if title else body)[:MAXC]


def main():
    D.mkdir(exist_ok=True)
    for name, repo in SETS.items():
        try:
            corpus, queries = first(load_dataset(repo, "corpus")), first(load_dataset(repo, "queries"))
        except Exception:
            corpus = first(load_dataset(repo, data_files={"test": "corpus/*.parquet"}))
            queries = first(load_dataset(repo, data_files={"test": "queries/*.parquet"}))
        rel = {}
        for r in qrels_for(repo):
            if float(r["score"]) > 0:
                rel.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(float(r["score"]))
        qrows = [{"id": str(q["_id"]), "text": q["text"][:MAXC]} for q in queries if str(q["_id"]) in rel]
        crows = [{"id": str(c["_id"]), "text": text_of(c)} for c in corpus]
        dump(D / f"{name}_corpus.jsonl", crows)
        dump(D / f"{name}_queries.jsonl", qrows)
        json.dump(rel, open(D / f"{name}_qrels.json", "w"))
        print(f"{name}: {len(crows)} documents, {len(qrows)} queries")

    random.seed(7)
    nq = load_dataset("sentence-transformers/natural-questions", split="train")
    rows = []
    for i in random.sample(range(len(nq)), 5000):
        rows += [{"text": nq[i]["query"][:MAXC]}, {"text": nq[i]["answer"][:MAXC]}]
    dump(D / "train.jsonl", rows)
    print(f"train: {len(rows)} texts")


if __name__ == "__main__":
    main()
