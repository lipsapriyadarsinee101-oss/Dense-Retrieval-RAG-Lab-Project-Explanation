"""Retrieval-augmented generation with numbered citations.

    python -m retrieval_lab.rag --dataset mteb/scifact --question "Does vitamin D reduce fracture risk?"

Retrieves passages with hybrid BM25 + dense search, asks a small instruction
model to answer only from them and cite [n], then checks the citations.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .bm25 import BM25Index
from .data import load
from .dense import BiEncoder, dense_search
from .fusion import reciprocal_rank_fusion

PROMPT = (
    "Answer the question using only the numbered passages. Cite every claim with the passage "
    "number in square brackets, e.g. [2]. If the passages do not contain the answer, say so.\n\n"
    "{passages}\n\nQuestion: {question}\nAnswer:"
)


class HybridRetriever:
    def __init__(self, corpus: dict[str, str], model_path: str, device, cache: Path | None = None):
        self.corpus, self.doc_ids, self.device = corpus, list(corpus), device
        self.bm25 = BM25Index(corpus)
        self.encoder = BiEncoder(model_path).to(device)
        if cache and cache.exists():
            self.doc_emb = torch.load(cache)
        else:
            self.doc_emb = self.encoder.encode([corpus[d] for d in self.doc_ids], is_query=False, device=device)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.doc_emb, cache)

    def retrieve(self, question: str, k: int = 5) -> list[str]:
        sparse = self.bm25.search({"q": question}, k=100)
        q_emb = self.encoder.encode([question], is_query=True, device=self.device)
        dense = dense_search(q_emb, ["q"], self.doc_emb, self.doc_ids, k=100)
        fused = reciprocal_rank_fusion([sparse, dense], top=k)["q"]
        return list(fused)


def check_citations(answer: str, n_passages: int) -> dict:
    cited = sorted({int(c) for c in re.findall(r"\[(\d+)\]", answer)})
    return {"cited": cited, "invalid": [c for c in cited if not 1 <= c <= n_passages], "has_citation": bool(cited)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="mteb/scifact")
    p.add_argument("--retriever", default="intfloat/multilingual-e5-small", help="base or fine-tuned model folder")
    p.add_argument("--generator", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--question", required=True)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load(args.dataset, "test")
    cache = Path(".cache") / f"{Path(args.dataset).name}-{Path(args.retriever).name}.pt"
    retriever = HybridRetriever(data.corpus, args.retriever, device, cache)
    doc_ids = retriever.retrieve(args.question, k=args.k)
    passages = "\n".join(f"[{i}] {data.corpus[d][:700]}" for i, d in enumerate(doc_ids, 1))

    tok = AutoTokenizer.from_pretrained(args.generator)
    lm = AutoModelForCausalLM.from_pretrained(args.generator).to(device)
    msgs = [{"role": "user", "content": PROMPT.format(passages=passages, question=args.question)}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
    out = lm.generate(inputs, max_new_tokens=256, do_sample=False)
    answer = tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True).strip()

    print("Retrieved:", ", ".join(doc_ids))
    print("\nAnswer:\n" + answer)
    print("\nCitation check:", check_citations(answer, len(doc_ids)))


if __name__ == "__main__":
    main()
