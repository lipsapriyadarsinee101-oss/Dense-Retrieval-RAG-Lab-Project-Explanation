"""Contrastive (InfoNCE) loss with in-batch, hard and cross-GPU negatives."""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


class _AllGather(torch.autograd.Function):
    """All-gather that is differentiable.

    Forward concatenates every rank's tensor. Backward sums the gradient each
    rank computed for the full tensor and hands back this rank's slice, so the
    local embeddings receive gradient from the losses on every GPU.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.rank, ctx.n = dist.get_rank(), x.size(0)
        out = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x.contiguous())
        return torch.cat(out, dim=0)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        grad = grad.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad[ctx.rank * ctx.n:(ctx.rank + 1) * ctx.n]


def _gather(x: torch.Tensor) -> torch.Tensor:
    return _AllGather.apply(x)


def info_nce(q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor | None = None,
             temperature: float = 0.05, cross_device: bool = True) -> torch.Tensor:
    """q[i] should score pos[i] above every other candidate.

    Candidates = all positives + all hard negatives in the batch. With
    cross_device=True under DDP, candidates are gathered from every GPU, so the
    number of negatives grows with world size at no extra encoding cost.
    """
    offset = 0
    if cross_device and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        offset = dist.get_rank() * q.size(0)
        pos = _gather(pos)
        neg = _gather(neg) if neg is not None else None
    docs = pos if neg is None else torch.cat([pos, neg], dim=0)
    logits = (q @ docs.T).float() / temperature
    labels = torch.arange(q.size(0), device=q.device) + offset
    return F.cross_entropy(logits, labels)
