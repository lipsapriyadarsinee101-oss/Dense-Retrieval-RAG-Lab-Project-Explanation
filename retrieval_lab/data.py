"""Load retrieval datasets in BEIR/MTEB format (corpus, queries, qrels)."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievalData:
    corpus: dict[str, str]              # doc_id -> text
    queries: dict[str, str]             # query_id -> text
    qrels: dict[str, dict[str, int]]    # query_id -> {doc_id: relevance}


def _finalize(corpus, queries, qrels) -> RetrievalData:
    # keep only queries that have at least one judged relevant document
    queries = {q: t for q, t in queries.items() if q in qrels}
    return RetrievalData(corpus, queries, qrels)


def load_hf(name: str = "mteb/scifact", split: str = "test") -> RetrievalData:
    """Load an MTEB retrieval dataset from the Hugging Face Hub."""
    from datasets import load_dataset

    corpus_ds = load_dataset(name, "corpus", split="corpus")
    queries_ds = load_dataset(name, "queries", split="queries")
    qrels_ds = load_dataset(name, "default", split=split)

    corpus = {str(r["_id"]): f"{r.get('title') or ''} {r['text']}".strip() for r in corpus_ds}
    queries = {str(r["_id"]): r["text"] for r in queries_ds}
    qrels: dict[str, dict[str, int]] = {}
    for r in qrels_ds:
        if int(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    return _finalize(corpus, queries, qrels)


def load_local(data_dir: str | Path, split: str = "test") -> RetrievalData:
    """Load a local BEIR folder: corpus.jsonl, queries.jsonl, qrels/<split>.tsv.

    Works for any language, e.g. a German or cross-lingual test set you build yourself.
    """
    d = Path(data_dir)
    corpus, queries = {}, {}
    with open(d / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            corpus[str(r["_id"])] = f"{r.get('title') or ''} {r['text']}".strip()
    with open(d / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            queries[str(r["_id"])] = r["text"]
    qrels: dict[str, dict[str, int]] = {}
    with open(d / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for qid, did, score in reader:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return _finalize(corpus, queries, qrels)


def load(dataset: str, split: str) -> RetrievalData:
    """`dataset` is either a local folder or a Hugging Face dataset name."""
    return load_local(dataset, split) if Path(dataset).is_dir() else load_hf(dataset, split)
