# Dense Retrieval & RAG Lab

Train and evaluate a **multilingual dense retriever** with contrastive learning in PyTorch, compare it against a **BM25 sparse baseline** and a **hybrid** of both, and use the result for **retrieval-augmented generation with citations**.

The goal is a small, honest search-quality loop: change the model or the training data → measure nDCG / MRR / Recall on judged queries → keep what helps.

## What's inside

| Component | File | What it does |
|---|---|---|
| Sparse retrieval | `retrieval_lab/bm25.py` | BM25 with a Unicode-aware tokenizer (works beyond English) |
| Dense retrieval | `retrieval_lab/dense.py` | Bi-encoder (default `intfloat/multilingual-e5-small`), mean pooling, normalized embeddings, exact top-k search |
| Contrastive training | `retrieval_lab/train.py`, `losses.py` | InfoNCE with in-batch negatives + BM25-mined hard negatives; DDP multi-GPU with **negatives gathered across GPUs**; bf16 autocast, gradient checkpointing, linear warmup |
| Representation fusion | `retrieval_lab/fusion.py` | Reciprocal Rank Fusion and weighted min-max score fusion of sparse + dense |
| Evaluation | `retrieval_lab/metrics.py`, `benchmark.py` | nDCG@10, MRR@10, Recall@100 (trec_eval definitions), one table for all systems |
| RAG | `retrieval_lab/rag.py` | Hybrid retrieval → small instruction LLM answers only from numbered passages → citation check |
| Tests + CI | `tests/`, `.github/workflows/` | Metric and fusion unit tests, loss sanity checks, and an end-to-end train + benchmark run on a tiny local model |

## Quickstart

```bash
pip install -r requirements.txt
pytest -q                      # runs offline in under a minute

# 1. Baselines: BM25, dense (zero-shot) and hybrid on SciFact
python -m retrieval_lab.benchmark --dataset mteb/scifact

# 2. Fine-tune the retriever on the SciFact train split
python -m retrieval_lab.train --dataset mteb/scifact --output runs/e5-scifact --epochs 3

#    ...or on several GPUs (negatives are shared across GPUs)
torchrun --nproc_per_node=4 -m retrieval_lab.train --dataset mteb/scifact --output runs/e5-scifact

# 3. Compare everything on the test split
python -m retrieval_lab.benchmark --dataset mteb/scifact --finetuned runs/e5-scifact

# 4. Ask a question with grounded, cited answers
python -m retrieval_lab.rag --dataset mteb/scifact --retriever runs/e5-scifact \
    --question "Does vitamin D supplementation reduce fracture risk?"
```

Any dataset in BEIR format works: pass a Hugging Face MTEB dataset name or a local folder with `corpus.jsonl`, `queries.jsonl` and `qrels/<split>.tsv`. This is how to evaluate German or cross-lingual retrieval (e.g. German queries against an English corpus).

## Results

Fill this table with the output of `results/scifact.md` after running steps 1–3 (numbers come only from your own runs).

| System | nDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| BM25 | | | |
| Dense, zero-shot (multilingual-e5-small) | | | |
| Dense, fine-tuned | | | |
| Hybrid RRF (BM25 + fine-tuned dense) | | | |

## Design notes

- **Hard negatives with a gap.** Negatives come from BM25 ranks 4–50 that are not judged relevant. The top 3 hits are skipped because unjudged relevant documents (false negatives) concentrate there.
- **Cross-device negatives.** Under DDP, positives and hard negatives are all-gathered with gradients (`torch.distributed.nn`), so each query is contrasted against `2 × batch × GPUs` documents for the cost of encoding its own batch.
- **One forward pass per step.** Queries and documents are padded into a single batch so DDP sees exactly one forward per backward.
- **Temperature.** 0.05 by default. Lower values sharpen the softmax and push harder on near-miss negatives.
- **Why RRF.** BM25 scores and cosine similarities live on different scales; rank-based fusion avoids calibrating them.

## Next steps

- Cross-lingual alignment: train on translated query–passage pairs and evaluate German→English retrieval
- Learned sparse retrieval (SPLADE-style) as a third run to fuse
- Cross-encoder reranker on the top 100
- FAISS / HNSW index and latency vs. quality trade-offs
- RAG evaluation: answer faithfulness to cited passages
