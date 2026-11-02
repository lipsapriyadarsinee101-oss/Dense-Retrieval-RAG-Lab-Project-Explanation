"""IR evaluation metrics: nDCG@k, MRR@k, Recall@k (same definitions as trec_eval / BEIR)."""
from __future__ import annotations

import math

Run = dict[str, dict[str, float]]    # query_id -> {doc_id: score}
Qrels = dict[str, dict[str, int]]    # query_id -> {doc_id: relevance}


def _ranked(scores: dict[str, float], k: int) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    relevant = {d for d, r in rels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def evaluate(run: Run, qrels: Qrels, ndcg_k: int = 10, mrr_k: int = 10, recall_k: int = 100) -> dict[str, float]:
    """Average metrics over all judged queries; a query missing from the run scores 0."""
    totals = {f"nDCG@{ndcg_k}": 0.0, f"MRR@{mrr_k}": 0.0, f"Recall@{recall_k}": 0.0}
    for qid, rels in qrels.items():
        ranked = _ranked(run.get(qid, {}), max(ndcg_k, mrr_k, recall_k))
        totals[f"nDCG@{ndcg_k}"] += ndcg_at_k(ranked, rels, ndcg_k)
        totals[f"MRR@{mrr_k}"] += mrr_at_k(ranked, rels, mrr_k)
        totals[f"Recall@{recall_k}"] += recall_at_k(ranked, rels, recall_k)
    n = max(len(qrels), 1)
    return {m: v / n for m, v in totals.items()}
