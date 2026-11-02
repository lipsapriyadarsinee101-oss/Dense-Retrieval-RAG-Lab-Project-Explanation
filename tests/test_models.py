"""Tests that need torch but no internet: a tiny randomly initialised BERT is built locally."""
import json

import pytest
import torch

from retrieval_lab.bm25 import BM25Index, tokenize
from retrieval_lab.losses import info_nce


def test_tokenizer_is_unicode_aware():
    assert tokenize("Größe der Straße, naïve café") == ["größe", "der", "straße", "naïve", "café"]


def test_bm25_finds_lexical_match():
    idx = BM25Index({"d1": "cats chase mice", "d2": "stock market news", "d3": "dogs bark loudly"})
    run = idx.search({"q": "market"}, k=2)
    assert max(run["q"], key=run["q"].get) == "d2"


def test_info_nce_low_when_aligned_high_when_shuffled():
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    neg = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    aligned = info_nce(q, q.clone(), neg, temperature=0.05)
    shuffled = info_nce(q, q.roll(1, 0), neg, temperature=0.05)
    assert aligned < 0.1 < shuffled


@pytest.fixture
def tiny_setup(tmp_path):
    from transformers import BertConfig, BertModel, BertTokenizerFast

    words = "cats chase mice stock market news dogs bark loudly what do query passage : the".split()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    model_dir = tmp_path / "tiny-bert"
    model_dir.mkdir()
    (model_dir / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt")).save_pretrained(model_dir)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64, max_position_embeddings=64)
    BertModel(cfg).save_pretrained(model_dir)

    data_dir = tmp_path / "toy"
    (data_dir / "qrels").mkdir(parents=True)
    docs = {"d1": "cats chase mice", "d2": "stock market news", "d3": "dogs bark loudly", "d4": "the market"}
    qs = {"q1": "what do cats chase", "q2": "market news", "q3": "dogs"}
    rels = [("q1", "d1"), ("q2", "d2"), ("q3", "d3")]
    (data_dir / "corpus.jsonl").write_text("\n".join(json.dumps({"_id": k, "title": "", "text": v}) for k, v in docs.items()))
    (data_dir / "queries.jsonl").write_text("\n".join(json.dumps({"_id": k, "text": v}) for k, v in qs.items()))
    for split in ("train", "test"):
        (data_dir / "qrels" / f"{split}.tsv").write_text("query-id\tcorpus-id\tscore\n" + "\n".join(f"{q}\t{d}\t1" for q, d in rels))
    return model_dir, data_dir, tmp_path


def test_encoder_outputs_unit_vectors(tiny_setup):
    from retrieval_lab.dense import BiEncoder

    model_dir, _, _ = tiny_setup
    emb = BiEncoder(str(model_dir)).encode(["cats chase mice", "dogs"], is_query=False)
    assert emb.shape == (2, 32)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_train_and_benchmark_end_to_end(tiny_setup, monkeypatch):
    from retrieval_lab import benchmark, train

    model_dir, data_dir, tmp = tiny_setup
    out = tmp / "ft"
    train.main(["--dataset", str(data_dir), "--model", str(model_dir), "--output", str(out),
                "--batch-size", "2", "--epochs", "2", "--neg-depth", "4", "--max-len", "16"])
    assert (out / "config.json").exists() and (out / "train_args.json").exists()

    monkeypatch.chdir(tmp)
    benchmark.main(["--dataset", str(data_dir), "--base-model", str(model_dir), "--finetuned", str(out), "--k", "4"])
    results = json.loads((tmp / "results" / "toy.json").read_text())
    assert "BM25" in results and "Dense (fine-tuned)" in results
    assert all(0.0 <= v <= 1.0 for r in results.values() for v in r.values())
