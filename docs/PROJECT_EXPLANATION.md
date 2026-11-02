# Project Explanation

This project trains a search model that finds the right documents for a question, measures how good it is, and uses it to answer questions with citations. It follows the core loop of search-quality work: change the model or training data, measure, keep what helps.

## The pipeline

The project runs in five stages. Each stage is one command, and each produces something the next stage uses.

1. **Load data.** A retrieval dataset has three parts: a corpus (the documents), queries (the questions) and qrels (human judgements of which documents answer which query). The default is SciFact: about 5,000 scientific abstracts, with separate train and test queries.
2. **Baselines.** Two retrievers search the corpus for every test query:
   - BM25 matches words (sparse retrieval).
   - A pretrained multilingual neural model, multilingual-e5-small, matches meaning (dense retrieval).
3. **Fine-tune.** The neural model is trained on the train queries with contrastive learning, so it learns what "relevant" means in this domain.
4. **Evaluate and fuse.** Every system is scored on the test queries with nDCG@10, MRR@10 and Recall@100. A hybrid that merges the BM25 and neural rankings is scored too. All results land in one table.
5. **Answer questions (RAG).** For a new question, the hybrid retriever picks the top 5 passages. A small language model writes an answer using only those passages and cites them as [1], [2]. A checker confirms every citation points to a real passage.

The test set never touches training, so the results table shows honestly whether fine-tuning and fusion helped.

## Key concepts

**BM25 (sparse retrieval).** Scores a document by how many query words it contains. Rare words count more, and the score is adjusted so long documents don't win just by being long. It is fast and precise on exact terms, such as names and codes, but misses synonyms: "car" won't match "automobile". The tokenizer handles accents and umlauts, so it also works on German.

**Dense retrieval (bi-encoder).** A transformer turns every text into a vector of numbers, its embedding. Similar meanings get nearby vectors, so search becomes "find the document vectors closest to the query vector". "Bi-encoder" means queries and documents are encoded separately, so all documents can be encoded once in advance.

- **Mean pooling.** The model outputs one vector per token. These are averaged into a single vector, ignoring padding, then scaled to length 1 so that the dot product equals cosine similarity.
- **Prefixes.** E5 models expect "query: " and "passage: " in front of the text, so the model knows which side it is encoding.

**Contrastive learning (InfoNCE loss).** Training shows the model a query, its correct document, and many wrong documents. The loss is a softmax over all candidates, pushing the correct one to the top. It is cross-entropy where the "class" is the correct document. Every other document in the batch serves as a free negative (in-batch negatives).

**Hard negatives.** Random wrong documents are too easy; the model learns little from them. So for each query the code takes BM25 results that look relevant but are not judged relevant. The top 3 BM25 hits are skipped, because unlabeled but truly relevant documents (false negatives) cluster there.

**Temperature (0.05).** Similarity scores are divided by this number before the softmax. A smaller temperature makes the model focus harder on near-miss negatives. Too small makes training unstable.

**Distributed training (DDP).** With `torchrun --nproc_per_node=4`, four copies of the model each process a different slice of the data and average their gradients after every step. On top of that:

- **Cross-GPU negatives.** Before computing the loss, each GPU all-gathers the document embeddings from all GPUs, so each query is compared against 4× as many negatives at no extra encoding cost. The custom `_AllGather` function in `losses.py` passes gradients back to each GPU's own slice.
- **One forward pass per step.** Queries and documents go through the model together, because DDP breaks if the model is called several times before one backward pass.
- **No pooler.** The [CLS] pooler is unused with mean pooling and would receive no gradient, which DDP rejects, so it is removed.

**Speed and memory options.**

- **bf16 autocast:** runs most maths in 16-bit for speed, keeping the loss in 32-bit for accuracy.
- **Gradient checkpointing:** recomputes activations during backward instead of storing them, trading compute for memory.
- **Linear warmup:** starts with a tiny learning rate so early gradients don't damage the pretrained weights.

**Hybrid retrieval (Reciprocal Rank Fusion).** BM25 scores and cosine scores are on different scales, so they can't simply be added. RRF uses ranks instead: each document gets 1/(60 + rank) from each system, and the totals are summed. Documents ranked well by both systems win, which helps because sparse and dense retrieval make different mistakes.

**Evaluation metrics.**

- **nDCG@10:** quality of the top 10, rewarding relevant documents more the higher they appear. 1.0 is a perfect ranking. This is the headline metric in BEIR and MTEB.
- **MRR@10:** 1 divided by the position of the first relevant result: how soon the user sees something useful.
- **Recall@100:** the share of all relevant documents found in the top 100. This matters when a reranker or RAG step reads the top results.

**RAG (retrieval-augmented generation).** The language model answers only from retrieved passages instead of from memory, which reduces made-up facts. Numbered citations let a reader verify each claim. The citation check flags answers with no citations or with citations to passages that don't exist.

## File-by-file guide

| File | What it does | Look for |
| --- | --- | --- |
| `retrieval_lab/data.py` | Loads corpus, queries and qrels from Hugging Face or a local BEIR folder | `load()` picks the source; queries without judgements are dropped |
| `retrieval_lab/metrics.py` | nDCG, MRR and Recall from scratch | Linear gain in nDCG, same as trec_eval; missing queries score 0 |
| `retrieval_lab/bm25.py` | BM25 index and search | `k1=0.9, b=0.4` (BEIR defaults); `argpartition` for fast top-k |
| `retrieval_lab/dense.py` | Bi-encoder model, encoding and exact search | Mean pooling, normalisation, pooler removed for DDP, length-sorted batches |
| `retrieval_lab/losses.py` | InfoNCE loss and the cross-GPU gather | `_AllGather.backward` sums gradients and returns this GPU's slice |
| `retrieval_lab/train.py` | Hard-negative mining and the training loop | `mine_triples()`, single forward per step, bf16, warmup, clipping, DDP setup |
| `retrieval_lab/fusion.py` | RRF and weighted score fusion | Rank-based vs min-max normalised scores |
| `retrieval_lab/benchmark.py` | Runs all systems and writes the results table | Writes `results/<dataset>.md` and `.json` |
| `retrieval_lab/rag.py` | Hybrid retrieval, answer generation, citation check | Passage-only prompt; cached embeddings |
| `tests/` | 11 automated tests | A tiny model is built offline to test training end to end |
| `.github/workflows/tests.yml` | Runs the tests on every push | Green tick on GitHub |

## Running it on a free Colab GPU

In Colab, choose Runtime → Change runtime type → T4 GPU, then run:

```bash
!git clone https://github.com/lipsapriyadarsinee101-oss/Dense-Retrieval-RAG-Lab-Project-Explanation.git
%cd Dense-Retrieval-RAG-Lab-Project-Explanation
!pip install -q transformers datasets rank_bm25
!pytest -q
!python -m retrieval_lab.benchmark --dataset mteb/scifact
!python -m retrieval_lab.train --dataset mteb/scifact --output runs/e5-scifact --epochs 3
!python -m retrieval_lab.benchmark --dataset mteb/scifact --finetuned runs/e5-scifact
!python -m retrieval_lab.rag --dataset mteb/scifact --retriever runs/e5-scifact --question "Does vitamin D reduce fracture risk?"
```

Copy the table from `results/scifact.md` into the Results section of the README.
