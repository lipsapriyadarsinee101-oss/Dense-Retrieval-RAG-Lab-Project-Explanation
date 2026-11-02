"""Contrastive fine-tuning of a dense retriever with BM25-mined hard negatives.

Single GPU / CPU:
    python -m retrieval_lab.train --dataset mteb/scifact --output runs/e5-scifact
Multi-GPU (DistributedDataParallel, negatives shared across GPUs):
    torchrun --nproc_per_node=4 -m retrieval_lab.train --dataset mteb/scifact --output runs/e5-scifact
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup

from .bm25 import BM25Index
from .data import RetrievalData, load
from .dense import BiEncoder
from .losses import info_nce


def mine_triples(data: RetrievalData, depth: int = 50, skip_top: int = 3, seed: int = 0) -> list[tuple[str, str, str]]:
    """(query, positive, hard negative) triples.

    Hard negatives are BM25 results that are not judged relevant. The top few
    BM25 hits are skipped because unjudged-but-relevant documents (false
    negatives) cluster there and would teach the model the wrong thing.
    """
    rng = random.Random(seed)
    run = BM25Index(data.corpus).search(data.queries, k=depth)
    all_ids = list(data.corpus)
    triples = []
    for qid, rels in data.qrels.items():
        if qid not in data.queries:
            continue
        ranked = [d for d, _ in sorted(run[qid].items(), key=lambda x: -x[1])]
        cands = [d for d in ranked[skip_top:] if d not in rels] or [d for d in all_ids if d not in rels]
        for pid in rels:
            triples.append((data.queries[qid], data.corpus[pid], data.corpus[rng.choice(cands)]))
    return triples


def setup_distributed() -> tuple[int, int, torch.device]:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)
        return dist.get_rank(), dist.get_world_size(), torch.device("cpu")
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="mteb/scifact", help="HF dataset name or local BEIR folder")
    p.add_argument("--train-split", default="train")
    p.add_argument("--model", default="intfloat/multilingual-e5-small")
    p.add_argument("--output", default="runs/finetuned")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32, help="per GPU")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--neg-depth", type=int, default=50)
    p.add_argument("--no-cross-device", action="store_true", help="only use negatives from the local GPU")
    p.add_argument("--grad-checkpointing", action="store_true", help="trade compute for memory")
    p.add_argument("--max-steps", type=int, default=-1, help="stop early (for smoke tests)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for tokenization")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rank, world, device = setup_distributed()
    torch.manual_seed(args.seed + rank)
    is_main = rank == 0

    data = load(args.dataset, args.train_split)
    triples = mine_triples(data, depth=args.neg_depth, seed=args.seed)  # deterministic: same on every rank
    if is_main:
        print(f"{len(triples)} training triples from {len(data.queries)} queries; world size {world}")

    model = BiEncoder(args.model, max_len=args.max_len).to(device)
    if args.grad_checkpointing:
        model.encoder.gradient_checkpointing_enable()
    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None) if world > 1 else model

    def collate(batch):
        qs, ps, ns = zip(*batch)
        # One forward pass for queries + docs: DDP expects a single forward per backward.
        q = model.tokenize(list(qs), is_query=True)
        d = model.tokenize(list(ps) + list(ns), is_query=False)
        return q, d

    sampler = DistributedSampler(triples, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
    loader = DataLoader(triples, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
                        collate_fn=collate, drop_last=True, num_workers=args.num_workers)

    total_steps = len(loader) * args.epochs if args.max_steps < 0 else args.max_steps
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(optim, int(args.warmup * total_steps), total_steps)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    step, t0 = 0, time.time()
    ddp_model.train()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for q, d in loader:
            q = {k: v.to(device, non_blocking=True) for k, v in q.items()}
            d = {k: v.to(device, non_blocking=True) for k, v in d.items()}
            # pad queries and docs to one length so they share a single forward pass
            L = max(q["input_ids"].size(1), d["input_ids"].size(1))
            pad = model.tokenizer.pad_token_id
            ids = torch.cat([torch.nn.functional.pad(q["input_ids"], (0, L - q["input_ids"].size(1)), value=pad),
                             torch.nn.functional.pad(d["input_ids"], (0, L - d["input_ids"].size(1)), value=pad)])
            mask = torch.cat([torch.nn.functional.pad(q["attention_mask"], (0, L - q["attention_mask"].size(1))),
                              torch.nn.functional.pad(d["attention_mask"], (0, L - d["attention_mask"].size(1)))])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                emb = ddp_model(ids, mask)
            B = q["input_ids"].size(0)
            qe, pe, ne = emb[:B], emb[B:2 * B], emb[2 * B:]
            loss = info_nce(qe, pe, ne, temperature=args.temperature, cross_device=not args.no_cross_device)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            step += 1
            if is_main and (step % 20 == 0 or step == 1):
                rate = step * B * world / (time.time() - t0)
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {rate:.1f} triples/s")
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    if is_main:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        model.encoder.save_pretrained(out)
        model.tokenizer.save_pretrained(out)
        (out / "train_args.json").write_text(json.dumps(vars(args), indent=2))
        print(f"saved to {out}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
