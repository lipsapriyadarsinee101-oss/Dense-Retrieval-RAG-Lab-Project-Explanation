import math

from retrieval_lab.fusion import reciprocal_rank_fusion, weighted_score_fusion
from retrieval_lab.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg_single_relevant_at_rank_2():
    assert math.isclose(ndcg_at_k(["d1", "d2", "d3"], {"d2": 1}, 10), 1 / math.log2(3))


def test_ndcg_perfect_and_graded():
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 10) == 1.0
    assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 10) < 1.0


def test_mrr_and_recall():
    assert mrr_at_k(["x", "y", "z"], {"z": 1}, 10) == 1 / 3
    assert mrr_at_k(["x", "y", "z"], {"z": 1}, 2) == 0.0
    assert recall_at_k(["a", "b", "c"], {"a": 1, "d": 1}, 3) == 0.5


def test_evaluate_counts_missing_queries_as_zero():
    run = {"q1": {"d1": 3.0, "d2": 2.0}}
    qrels = {"q1": {"d1": 1}, "q2": {"d9": 1}}
    res = evaluate(run, qrels, recall_k=10)
    assert res["nDCG@10"] == 0.5
    assert res["MRR@10"] == 0.5


def test_rrf_rewards_agreement():
    sparse = {"q": {"a": 10.0, "b": 9.0, "c": 1.0}}
    dense = {"q": {"b": 0.9, "c": 0.8, "a": 0.1}}
    fused = reciprocal_rank_fusion([sparse, dense])["q"]
    assert list(fused)[0] == "b"  # ranked 2nd and 1st beats 1st and 3rd


def test_weighted_fusion_uses_weights():
    r1 = {"q": {"a": 1.0, "b": 0.0}}
    r2 = {"q": {"a": 0.0, "b": 1.0}}
    assert list(weighted_score_fusion([r1, r2], [0.8, 0.2])["q"])[0] == "a"
    assert list(weighted_score_fusion([r1, r2], [0.2, 0.8])["q"])[0] == "b"
