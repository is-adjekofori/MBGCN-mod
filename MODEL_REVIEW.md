# MBGCN on KuaiLive — Correctness & Efficiency Review

I read the paper (both MBGCN and KuaiLive), all the code, and confirmed the dataset shape. Short version: **your item-item modification is mathematically sound and faithful to the paper — it is not what's breaking convergence.** The near-zero metrics are almost entirely a scale problem, and there is one genuine memory bug plus a couple of modeling concerns that are hurting you.

## 1. Is the item-item modification mathematically correct?

Yes. Here is the equivalence so you can trust it.

The paper's item-item propagation (§3.4, §3.5.2) is a mean-aggregation over co-behaviored items: `s_i^(1) = W_t · mean_{j∈N_t(i)}(q_j)`, and the co-interaction graph for behavior *t* is exactly `S = Rᵀ R` (items×items), where `R` is your `relation_dict[t]` user×item matrix.

Your modified code (`model.py:183-191`):

```python
item_graph_degree     = Rᵀ @ (R @ ones)               # = rowsum(Rᵀ R) = rowsum(S)
tmp_item_propagation  = (Rᵀ @ (R @ E)) / degree @ W   # = D⁻¹ S E W
```

`Rᵀ(R·1) = rowsum(RᵀR)` and `Rᵀ(R·E) = (RᵀR)E = S·E` are algebraic identities. So your bypass computes **`D⁻¹ S E W` with `S = RᵀR`** — identical to first materializing `item_graph = RᵀR` and normalizing. Downstream (`score2`, the `Mt` projection, the concat of `q ‖ propagated`) all matches the paper's `y₂(u,i)` term exactly. **No correctness loss.**

One caveat worth verifying against the *original* `.pth` you replaced: `S = RᵀR` keeps the **diagonal** (`S_ii` = item i's user-degree) and uses **weighted counts**. If the original precomputed `item_*.pth` was binarized or had its diagonal removed, your version differs slightly in normalization (self-loop weight ≈ `1/avg_user_degree`). For live-streaming where users touch many streamers, that self-weight is small, so it's not harmful — but it's the one place your result won't byte-match the original. Not a bug, just a note.

## 2. Why your metrics look like zero (this is the real story)

It's overwhelmingly **scale**, not a math error:

- **452,621 items, 23,772 users.** Expected Recall@80 for a *random* model ≈ 80/452,621 ≈ **1.8e-4**. Printed at 3-4 decimals that reads as `0.000`.
- Target signal is tiny and hyper-sparse: `train.txt` = 72,646 interactions → density 6.7e-6. `test.txt` = 6,533 interactions, so most users have **0** test items (they're correctly excluded from the metric denominator).
- Item space is **38× larger than Tmall** (11,953 items). At the same lr/epochs, convergence is dramatically slower.

Before hunting for a bug, disambiguate "genuinely 0" from "tiny":
1. Print `mgnn_weight` each epoch (you already do at `train.py:93`) — if it's `NaN`, that's a real divergence; if it's drifting smoothly, training is fine.
2. Print metrics with more precision, or log the raw `_sum`/`_cnt`, or temporarily add Recall@500/@1000.
3. Run the **MF baseline** on the same split — if MF also shows ~0, it's the item space, not MBGCN.
4. Sanity check `scores` on one test user: are they all near-equal, all NaN, or actually spread out?

I checked the usual exactly-zero culprits and they're clean: the train-mask (`train.py:152`) only masks *training* items, not test items; no division-by-zero in the metric aggregation; no NaN path from the `+1e-8` denominators even for zero-degree users/items.

## 3. Genuine issues I did find (prioritized)

**🔴 Memory bomb — `dataset.py:74-113` (`__calculate_user_behaviour`).** You call `.to_dense()` on each `[23772 × 452621]` relation matrix — that's **~43 GB per behavior**, done twice per behavior. On CPU it survives only via swap (catastrophically slow); it will OOM a GPU instantly. This is very likely a big part of your "everything is slow" pain. Fix — never densify, use spmv:
```python
ones_i = torch.ones(self.item_num, 1)
ones_u = torch.ones(self.user_num, 1)
user_deg = torch.sparse.mm(R, ones_i)        # [U,1]  (or R @ ones_i)
item_deg = torch.sparse.mm(R.t(), ones_u)    # [I,1]
```
Mathematically identical, ~0 GB.

**🟠 `negative.txt` as a positive behavior (modeling correctness).** It's your largest signal (3.9M edges) and MBGCN has no notion of a *negative* edge — every relation graph is aggregated as "these items reflect similar preference." Treating "disliked" as co-interaction propagates embeddings *toward* each other, injecting wrong signal, and it dominates the `S=RᵀR` co-occurrence. I'd **drop it from `relation`** (keep it only for BPR negative sampling if anything). Try `relation='gift,comment,like,click'` and compare. This alone could be suppressing your metrics.

**🟡 `item_graph_degree` recomputed every minibatch (`model.py:183`).** It depends only on `R`, not on embeddings — it's constant. Precompute once in `__init__` and reuse. Cheap win. (The full `D⁻¹SE` propagation genuinely must be recomputed each step for gradients — that part you can't avoid.)

**🟡 Node dropout is inconsistent.** In `forward` you build `tmp_relation_matrix` with dropped values but the item-item propagation (`model.py:183-191`) uses the **undropped** `relation_mat`. Minor, and arguably matches the original (item graph was precomputed), but worth knowing.

**⚪ Dead/oversized code:** `__decode_weight` (`model.py:125`) calls `nn.softmax` which doesn't exist — but the method is never called (forward computes α directly per paper Eq. 2, correctly). Also `MBGCN.sh` passes `mgnn_weight4` twice to reach 5 weights for KuaiLive's 5 relations — it works but is fragile.

**⚪ Operational:** `sample_file/` doesn't exist in `dataset/nw_interaction/`. `newit()` increments the counter each epoch and reads `sample_{cnt}.txt`, so you need `max_epoch` presampled files or training crashes mid-run. Make sure `sample.py --max_epoch` ≥ your epoch count.

## 4. Efficiency / memory summary

Ranked by impact for your hardware:
1. **Kill the `.to_dense()`** (above) — the single biggest memory fix.
2. **Drop `negative`** — cuts the heaviest co-occurrence graph *and* likely improves metrics.
3. **Cache `item_graph_degree`** in the dataset once.
4. Consider **fewer epochs of presampling + higher lr / lr warmup**, or reduce candidate space for eval (the `[512 × 452621]` score tensor is ~0.9 GB/test-batch — lower `test_batch_size` if eval OOMs).
5. The per-batch full-graph propagation is inherent to this architecture (LightGCN/NGCF do the same); there's no *correct* shortcut without changing the gradient. Don't over-optimize it.

Beyond that, I wouldn't invent further "optimizations." The architecture is small (32–64 dim embeddings); your bottlenecks are the densification bug and the sheer item count, not the GCN math.

## Dataset facts (for reference)

- `user_num = 23,772`, `item_num = 452,621` (`dataset/nw_interaction/data_size.txt`)
- Interaction counts: gift 45,355 · comment 115,378 · like 105,048 · click 2,424,082 · negative 3,898,867
- Target split: train 72,646 · validation 6,360 · test 6,533
