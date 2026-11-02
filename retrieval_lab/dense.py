"""Dense bi-encoder retriever in PyTorch (mean pooling + L2-normalised embeddings)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer


class BiEncoder(nn.Module):
    """Shared-weight encoder for queries and documents.

    Defaults follow the E5 convention ("query: " / "passage: " prefixes), which
    lets one multilingual model embed queries and documents in ~100 languages.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", max_len: int = 256,
                 query_prefix: str = "query: ", doc_prefix: str = "passage: "):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        # We mean-pool token states, so the [CLS] pooler is unused. Removing it
        # avoids DDP's "parameters did not receive grad" error and saves memory.
        if getattr(self.encoder, "pooler", None) is not None:
            self.encoder.pooler = None
        self.max_len, self.query_prefix, self.doc_prefix = max_len, query_prefix, doc_prefix

    def tokenize(self, texts: list[str], is_query: bool) -> dict[str, torch.Tensor]:
        prefix = self.query_prefix if is_query else self.doc_prefix
        return self.tokenizer([prefix + t for t in texts], padding=True, truncation=True,
                              max_length=self.max_len, return_tensors="pt")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(pooled, dim=-1)

    @torch.no_grad()
    def encode(self, texts: list[str], is_query: bool, batch_size: int = 128,
               device: str | torch.device = "cpu") -> torch.Tensor:
        self.eval()
        out = []
        # sort by length so batches pad less (a cheap but real speed-up)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        use_amp = torch.device(device).type == "cuda"
        for s in range(0, len(texts), batch_size):
            batch = [texts[i] for i in order[s:s + batch_size]]
            enc = self.tokenize(batch, is_query)
            toks = {k: enc[k].to(device) for k in ("input_ids", "attention_mask")}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out.append(self(**toks).float().cpu())
        emb = torch.cat(out)
        restored = torch.empty_like(emb)
        restored[torch.tensor(order)] = emb
        return restored


def dense_search(query_emb: torch.Tensor, query_ids: list[str], doc_emb: torch.Tensor,
                 doc_ids: list[str], k: int = 100, chunk: int = 256) -> dict[str, dict[str, float]]:
    """Exact inner-product search (embeddings are normalised, so this is cosine)."""
    run = {}
    k = min(k, doc_emb.size(0))
    for s in range(0, query_emb.size(0), chunk):
        scores = query_emb[s:s + chunk] @ doc_emb.T
        top_s, top_i = scores.topk(k, dim=1)
        for row, qid in enumerate(query_ids[s:s + chunk]):
            run[qid] = {doc_ids[i]: float(v) for v, i in zip(top_s[row].tolist(), top_i[row].tolist())}
    return run
