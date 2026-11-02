"""Hybrid retrieval: fuse sparse and dense runs."""
from __future__ import annotations

Run = dict[str, dict[str, float]]


def reciprocal_rank_fusion(runs: list[Run], k: int = 60, top: int = 100) -> Run:
    """RRF (Cormack et al., 2009): score(d) = sum_r 1 / (k + rank_r(d)).

    Rank-based, so it needs no score calibration between BM25 and cosine scores.
    """
    fused: Run = {}
    for qid in set().union(*runs):
        scores: dict[str, float] = {}
        for run in runs:
            ranked = sorted(run.get(qid, {}).items(), key=lambda x: (-x[1], x[0]))
            for rank, (doc, _) in enumerate(ranked, start=1):
                scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
        fused[qid] = dict(sorted(scores.items(), key=lambda x: -x[1])[:top])
    return fused


def weighted_score_fusion(runs: list[Run], weights: list[float], top: int = 100) -> Run:
    """Min-max normalise each run per query, then take a weighted sum."""
    fused: Run = {}
    for qid in set().union(*runs):
        scores: dict[str, float] = {}
        for run, w in zip(runs, weights):
            s = run.get(qid, {})
            if not s:
                continue
            lo, hi = min(s.values()), max(s.values())
            for doc, v in s.items():
                norm = (v - lo) / (hi - lo) if hi > lo else 1.0
                scores[doc] = scores.get(doc, 0.0) + w * norm
        fused[qid] = dict(sorted(scores.items(), key=lambda x: -x[1])[:top])
    return fused
