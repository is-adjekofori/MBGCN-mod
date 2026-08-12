"""ATRank model in PyTorch (ID-only object embedding, KuaiLive).

Faithful port of reference/atrank_upstream/{atrank,multi}/model.py:
  history element = proj( concat[ id_emb, action_emb, onehot(time_bucket) ] )
  self-attention  = num_blocks of (multihead self-attn + position-wise FFN),
                    residual + layernorm, computed ONCE per user
  vanilla attn    = num_blocks of (target-as-query attn over history + FFN),
                    re-run per candidate (cheap) -> context vector
  score(cand)     = item_bias[cand] + dot( context, target_emb )

Differences from upstream, all deliberate and documented:
  * side features (shop/cate/brand) dropped -> ID-only, for a fair architecture
    comparison with the ID-only MBGCN.
  * loss is BPR pairwise (shared with MBGCN via loss.bprloss), not the paper's
    point-wise sigmoid CE, so the loss is held constant across the two models.
  * ranking metric is the shared sampled Recall/NDCG/MRR@k, not the paper's AUC.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import N_ACTIONS, N_TIME_BUCKETS

NEG_INF = -(2.0 ** 32) + 1.0


class MultiHeadAttention(nn.Module):
    """Upstream multihead_attention: relu Q/K/V projections, scaled dot-product,
    key + query masking, dropout, residual, layernorm. num_heads = K semantic
    spaces summing to `hidden`."""

    def __init__(self, hidden, num_heads, dropout):
        super().__init__()
        assert hidden % num_heads == 0
        self.h = num_heads
        self.d = hidden // num_heads
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden)

    def _split(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d).transpose(1, 2)  # [B,h,T,d]

    def forward(self, queries, keys, key_mask, query_mask):
        # queries [B,Tq,H]; keys [B,Tk,H]; masks [B,T] bool (True = valid)
        Q = self._split(F.relu(self.q(queries)))
        K = self._split(F.relu(self.k(keys)))
        V = self._split(F.relu(self.v(keys)))
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d ** 0.5)  # [B,h,Tq,Tk]
        km = key_mask[:, None, None, :]                                  # [B,1,1,Tk]
        scores = scores.masked_fill(~km, NEG_INF)
        attn = torch.softmax(scores, dim=-1)
        attn = attn * query_mask[:, None, :, None].to(attn.dtype)        # zero pad rows
        attn = self.drop(attn)
        out = torch.matmul(attn, V)                                      # [B,h,Tq,d]
        B, _, Tq, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.h * self.d)
        out = out + queries                                              # residual
        return self.ln(out)


class FeedForward(nn.Module):
    """Position-wise FFN (conv1d k=1 == Linear), inner = hidden//4, residual+LN."""

    def __init__(self, hidden):
        super().__init__()
        inner = max(1, hidden // 4)
        self.fc1 = nn.Linear(hidden, inner)
        self.fc2 = nn.Linear(inner, hidden)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return self.ln(h + x)


class ATRank(nn.Module):
    def __init__(self, item_num, id_dim=64, action_dim=64, hidden=128,
                 num_heads=8, num_blocks=1, dropout=0.1, reg=5e-5):
        super().__init__()
        self.item_num = item_num
        self.reg = reg
        self.item_emb = nn.Embedding(item_num, id_dim)
        self.item_bias = nn.Embedding(item_num, 1)
        self.action_emb = nn.Embedding(N_ACTIONS, action_dim)
        self.proj = nn.Linear(id_dim + action_dim + N_TIME_BUCKETS, hidden)
        self.id_dim, self.action_dim = id_dim, action_dim
        self.self_mha = nn.ModuleList(
            [MultiHeadAttention(hidden, num_heads, dropout) for _ in range(num_blocks)])
        self.self_ffn = nn.ModuleList([FeedForward(hidden) for _ in range(num_blocks)])
        self.van_mha = nn.ModuleList(
            [MultiHeadAttention(hidden, num_heads, dropout) for _ in range(num_blocks)])
        self.van_ffn = nn.ModuleList([FeedForward(hidden) for _ in range(num_blocks)])
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.action_emb.weight, std=0.01)

    def load_mf_item(self, mf_dir, normalize=True):
        """Warm-start item_emb from the SAME MF pretrain MBGCN uses (init parity).

        Reads <mf_dir>/model.pkl (produced by MF_KuaiLive.sh), takes its
        `item_embedding` [item_num, id_dim] over the identical streamer-id space,
        L2-normalizes it exactly as MBGCN does, and copies it into item_emb. MF's
        user_embedding is ignored (ATRank has no user vector).
        """
        import os
        data = torch.load(os.path.join(mf_dir, "model.pkl"), map_location="cpu",
                          weights_only=False)
        w = data["item_embedding"]
        if not torch.is_tensor(w):
            w = torch.as_tensor(w)
        if normalize:
            w = F.normalize(w)
        if w.shape != self.item_emb.weight.shape:
            raise ValueError(
                f"MF item_embedding {tuple(w.shape)} != ATRank item_emb "
                f"{tuple(self.item_emb.weight.shape)} (check embedding_size/id_dim)")
        with torch.no_grad():
            self.item_emb.weight.copy_(w)

    # ---- embeddings ---------------------------------------------------------
    def embed_history(self, streamer, action, timebucket):
        ide = self.item_emb(streamer)                       # [B,L,id]
        ae = self.action_emb(action)                        # [B,L,act]
        te = F.one_hot(timebucket, N_TIME_BUCKETS).to(ide.dtype)  # [B,L,16]
        return self.proj(torch.cat([ide, ae, te], dim=-1))  # [B,L,H]

    def embed_target(self, cand):
        # target has no action / no time -> zero-pad those channels (upstream pad)
        ide = self.item_emb(cand)                           # [B,C,id]
        z_act = ide.new_zeros(ide.shape[:-1] + (self.action_dim,))
        z_time = ide.new_zeros(ide.shape[:-1] + (N_TIME_BUCKETS,))
        return self.proj(torch.cat([ide, z_act, z_time], dim=-1))  # [B,C,H]

    # ---- attention ----------------------------------------------------------
    def self_encode(self, h, mask):
        enc = h
        for mha, ffn in zip(self.self_mha, self.self_ffn):
            enc = mha(enc, enc, mask, mask)
            enc = ffn(enc)
        return enc

    def vanilla_decode(self, enc, enc_mask, q):
        # q [B,C,H] as queries attending over history enc [B,L,H]
        qmask = torch.ones(q.shape[:2], dtype=torch.bool, device=q.device)
        dec = q
        for mha, ffn in zip(self.van_mha, self.van_ffn):
            dec = mha(dec, enc, enc_mask, qmask)
            dec = ffn(dec)
        return dec                                           # [B,C,H] context/cand

    def score_from_enc(self, enc, enc_mask, cand):
        q = self.embed_target(cand)                          # [B,C,H]
        dec = self.vanilla_decode(enc, enc_mask, q)          # [B,C,H]
        logits = (dec * q).sum(-1) + self.item_bias(cand).squeeze(-1)  # [B,C]
        return logits, q, dec

    # ---- training / eval ----------------------------------------------------
    def forward(self, streamer, action, timebucket, mask, pos, neg):
        enc = self.self_encode(self.embed_history(streamer, action, timebucket), mask)
        cand = torch.stack([pos, neg], dim=1)                # [B,2]
        logits, q, dec = self.score_from_enc(enc, mask, cand)  # [B,2]
        l2 = self.reg * (dec.pow(2).sum() + q.pow(2).sum())
        return logits, l2

    @torch.no_grad()
    def score_candidates(self, streamer, action, timebucket, mask, cand, chunk=1024):
        """Score [B, C] candidates; self-attention once, vanilla attn per chunk."""
        enc = self.self_encode(self.embed_history(streamer, action, timebucket), mask)
        outs = []
        for start in range(0, cand.shape[1], chunk):
            c = cand[:, start:start + chunk]
            logits, _, _ = self.score_from_enc(enc, mask, c)
            outs.append(logits)
        return torch.cat(outs, dim=1)                        # [B,C]
