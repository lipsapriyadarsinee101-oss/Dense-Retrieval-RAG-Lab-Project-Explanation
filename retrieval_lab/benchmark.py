"""Compare BM25, dense (base and fine-tuned) and hybrid retrieval on one test set.

    python -m retrieval_lab.benchmark --dataset mteb/scifact --finetuned runs/e5-scifact

Writes results/<name>.md and results/<name>.json with real numbers from your run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .bm25 import BM25Index
from .data import load
from .dense import BiEncoder, dense_search
from .fusion import reciprocal_rank_fusion
from .metrics import evaluate


def run_dense(model_path: str, data, k: int, device, batch_size: int):
    model = BiEncoder(model_path).to(device)
    doc_ids, query_ids = list(data.corpus), list(data.queries)
    t = time.time()
    d_emb = model.encode([data.corpus[d] for d in doc_ids], is_query=False, batch_size=batch_size, device=device)
    q_emb = model.encode([data.queries[q] for q in query_ids], is_query=True, batch_size=batch_size, device=device)
    run = dense_search(q_emb, query_ids, d_emb, doc_ids, k=k)
    return run, time.time() - t


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="mteb/scifact")
    p.add_argument("--split", default="test")
    p.add_argument("--base-model", default="intfloat/multilingual-e5-small")
    p.add_argument("--finetuned", default=None, help="folder written by retrieval_lab.train")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load(args.dataset, args.split)
    print(f"{len(data.corpus)} docs, {len(data.queries)} judged queries")

    runs, secs = {}, {}
    t = time.time()
    runs["BM25"] = BM25Index(data.corpus).search(data.queries, k=args.k)
    secs["BM25"] = time.time() - t
    runs["Dense (base)"], secs["Dense (base)"] = run_dense(args.base_model, data, args.k, device, args.batch_size)
    best_dense = "Dense (base)"
    if args.finetuned:
        runs["Dense (fine-tuned)"], secs["Dense (fine-tuned)"] = run_dense(args.finetuned, data, args.k, device, args.batch_size)
        best_dense = "Dense (fine-tuned)"
    runs[f"Hybrid RRF (BM25 + {best_dense})"] = reciprocal_rank_fusion([runs["BM25"], runs[best_dense]], top=args.k)

    results = {name: evaluate(run, data.qrels, recall_k=args.k) for name, run in runs.items()}
    metrics = list(next(iter(results.values())))
    lines = [f"Dataset: `{args.dataset}` ({args.split}), {len(data.queries)} queries, {len(data.corpus)} docs, device: {device}", "",
             "| System | " + " | ".join(metrics) + " | Time (s) |",
             "|---" * (len(metrics) + 2) + "|"]
    for name, res in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{res[m]:.4f}" for m in metrics) + f" | {secs.get(name, float('nan')):.1f} |")
    table = "\n".join(lines)
    print(table)

    out = Path("results")
    out.mkdir(exist_ok=True)
    stem = Path(args.dataset).name.replace("/", "_")
    (out / f"{stem}.md").write_text(table + "\n")
    (out / f"{stem}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
