"""Sparse lexical baseline: BM25 over a unicode-aware tokenizer."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, corpus: dict[str, str], k1: float = 0.9, b: float = 0.4):
        self.doc_ids = list(corpus)
        self.bm25 = BM25Okapi([tokenize(corpus[d]) for d in self.doc_ids], k1=k1, b=b)

    def search(self, queries: dict[str, str], k: int = 100) -> dict[str, dict[str, float]]:
        run = {}
        for qid, text in queries.items():
            scores = self.bm25.get_scores(tokenize(text))
            k_eff = min(k, len(scores))
            top = np.argpartition(-scores, k_eff - 1)[:k_eff]
            run[qid] = {self.doc_ids[i]: float(scores[i]) for i in top}
        return run
