# Thesis Write-up Changes Required to Reflect the Implemented Work

**Status as of 2026-08-11.** This document reconciles `Thesis_Chapter1_Draft2.md`
(the *planned* study) with what has actually been built and run for Part I
(MBGCN) and what is now being designed for Part II (ATRank). It is a **change
list**, not a rewrite — each item says what the draft currently claims, what the
implementation actually does, and the options for closing the gap. Decisions
marked **[YOUR CALL]** are scientific-framing choices only you can make; the rest
are factual corrections.

Sources of truth for "what was actually built": `docs/THESIS_MODIFICATIONS.md`
(§9–14, Part II), `model.py`, `dataset.py`, `loss.py`, the `MBGCN_KuaiLive.sh`
launcher, and `docs/MODEL_REVIEW.md`.

---

## Priority 1 — Load-bearing framing gaps (address before writing Ch. 3+)

### C1. Target behaviour: the draft says **gift**, the pipeline predicts **click** — [YOUR CALL]

**Where in the draft:** §1.4 Obj. III ("using virtual gift prediction as the
target behaviour"); §1.6 H0/H1/H2/H3 (all worded as *gift-through-rate* /
*gift prediction*); RQ2 and RQ4 ("virtual gifting", "a financial gift"); §1.8.1
("The target behaviour for all prediction tasks is virtual gift-giving");
Definition of Terms entries **GTR**, **Target Behaviour**, **Virtual Gift**,
**Auxiliary Behaviour**.

**What was actually built:** the target is **click**. Gift was attempted first
and abandoned because it is too sparse on KuaiLive to rank — only ~18 % of users
had a rank-able held-out gift and metrics collapsed to ~0.000. The re-target to
click is documented in `docs/THESIS_MODIFICATIONS.md` §9–10. All eval
infrastructure (`candidates.npz`, the leave-one-out split, the reported
Recall@20 = 0.565) is click-based.

**Why this is not just wording:** the study's whole scientific payoff (RQ2, RQ4,
H1–H3) rests on the target being the *sparse, high-effort, high-signal* action
with abundant auxiliaries propagating into it. Click is the **densest** behaviour,
not the sparsest — so "sparse target benefits from auxiliary propagation" and
"passive click vs. financial gift" (RQ4) lose their footing if click is the
target. This must be resolved explicitly, not silently.

**Options:**

- **(A) Adopt click as the target and re-frame the hypotheses around it.**
  Honest to the implementation and to the 0.565 result. Cost: H1–H3, RQ2, RQ4 and
  the GTR/Virtual-Gift framing all need rewording; the "sparse target" narrative
  becomes an "engagement prediction" narrative (which room/streamer a user will
  click into). Add a methodology subsection **"Target Behaviour Selection"** that
  presents the gift-sparsity evidence and justifies the pivot.
- **(B) Keep gift as the headline target, report it as a negative/infeasibility
  result, and use click as the operational task.** Keeps the draft's framing but
  requires you to actually run and report the gift experiment (even at ~0.000) as
  evidence, then argue click is the tractable proxy. More writing, more runs.
- **(C) Dual-target study:** gift where feasible (warm sub-population only) +
  click for the full population. Most work; strongest thesis if time allows.

**Recommendation:** (A) unless your supervisor specifically wants the gift result
preserved. It is the least self-contradictory and matches every number you
already have. Whichever you pick, **ATRank must use the same target as MBGCN** —
this is the fairness contract, non-negotiable.

### C2. Item granularity: the draft says **live room (ephemeral)**, the pipeline uses **streamer_id (persistent)** — [YOUR CALL]

**Where in the draft:** the "ephemeral item" thesis is *everywhere* — §1.1.4,
§1.2, §1.8.1, RQ3, H3, and the Definition of Terms entries **Ephemeral Item**,
**Live Room**. The draft repeatedly asserts the recommendation problem is
constructed *around live rooms* and that graph integrity degrades *as rooms
expire*.

**What was actually built:** item = **`streamer_id`** (452,621 persistent
streamers), **not** `live_id` (11.6 M ephemeral rooms). The ephemerality is not
discarded, though — it is captured in **candidate eligibility**: the 10,000
negatives per test case are *time-valid*, i.e. only streamers who were actually
live at the interaction's timestamp (reconstructed from `room.csv` lifecycles).
See `docs/THESIS_MODIFICATIONS.md` §11–12 and `memory/click-retarget-pipeline.md`.

**Why this matters:** modelling at streamer granularity is a defensible and
probably necessary choice (a room-level graph over 11.6 M ephemeral nodes barely
forms), but the draft currently implies room-level items. RQ3 and H3 ("as expired
live rooms accumulate, graph integrity degrades") describe a mechanism the
implementation does **not** instantiate at the item level.

**Options:**

- **(A) Re-frame the item as the streamer, with ephemerality relocated to
  candidate eligibility.** State plainly: items are streamers; the ephemeral
  nature of rooms enters through *time-valid candidate pools* (a streamer is only
  a valid recommendation at moments they are live). Re-word RQ3/H3 to test whether
  the *time-varying candidate set* (not a decaying node set) advantages sequential
  over graph models. This is the truthful description of what you built.
- **(B) Actually model room-level items.** Rebuild the entire pipeline at
  `live_id` granularity. Very large cost; likely infeasible given the graph
  sparsity that motivated the streamer aggregation in the first place. Not
  recommended.

**Recommendation:** (A). Add a methodology subsection **"Unit of
Recommendation"** making the streamer-item + time-valid-candidate choice explicit,
and revise RQ3/H3 to the candidate-eligibility mechanism. This is the single most
important correctness fix for the thesis's internal consistency.

### C3. Side features: the draft says features are **"included as numeric inputs"**, the models are **ID-only** — [DECIDED: ID-only]

**Where in the draft:** §1.9.4 ("These features are included in model training as
numeric inputs, but their contribution … cannot be independently interpreted");
implicitly §1.8.2.

**What was actually built / decided:** both MBGCN (as-built) and ATRank (as
planned) are **ID-only** — item = a single `streamer_id` embedding, no side
features. This is a deliberate **fairness decision** (see the fairness case in the
session log / `docs/THESIS_MODIFICATIONS.md`): side features are an *input*, not an
architectural property, so feeding them to one family would confound the
architecture comparison that RQ1 is meant to answer.

**KuaiLive does ship real side features** (confirmed from `streamer.csv` headers):
`live_operation_tag` (a genuine streamer category/tag), demographics, activity
counts, plus `onehot_feat0..6` (the 7 encrypted binaries), and per-room
`live_content_category` + `title_embeddings.npy` (745,576 × 128).

**Change required:** rewrite §1.9.4. It currently claims features *are* used and
frames the limitation as *opacity of used features*. The truthful statement is:

> For the primary architecture comparison, both models are evaluated in an
> **ID-only** configuration so that observed differences are attributable to
> architecture rather than feature engineering. Side information is therefore
> deliberately excluded from the head-to-head. A secondary **ablation** adds
> streamer-side features (notably `live_operation_tag` and room-title embeddings)
> to ATRank to probe cold-start behaviour; the interpretability of that ablation
> is limited by the encrypted, undisclosed semantics of several feature fields.

Move the "encrypted opacity" caveat so it scopes only the ablation, not the main
result.

---

## Priority 2 — Factual corrections (small edits, but wrong as written)

### C4. Implementation stack: **not** PyTorch Geometric

**Where:** §1.8.2 — "Graph construction for MBGCN uses the PyTorch Geometric
library (Fey & Lenssen, 2019)."

**Reality:** MBGCN is implemented in **plain PyTorch** with hand-rolled sparse
operations (`torch.sparse_coo_tensor`, sparse mm/sum in `dataset.py` and
`model.py`). There is **no PyG dependency**. The item–item propagation is a custom
`Rᵀ(R·E)` bypass, not a PyG message-passing layer (see `docs/MODEL_REVIEW.md` §1).

**Change:** delete the PyTorch Geometric claim and the Fey & Lenssen citation from
§1.8.2 (and the References list if it is not cited elsewhere). State: "implemented
in PyTorch using custom sparse-tensor graph operations."

### C5. Metric naming: **Recall@K** vs the draft's **HR@K**

**Where:** §1.8.2 and Definition of Terms (**HR@K**).

**Reality:** the code reports **Recall@{5,10,20}, NDCG@{5,10,20}, MRR@{5,10,20}**.
Under leave-one-out with exactly **one** held-out positive per user, Recall@K and
HR@K are **numerically identical** (both = 1 if the positive lands in the top-K).

**Change:** either (a) rename the reported metric to HR@K to match the draft, or
(b) keep Recall@K in the draft and add one sentence noting the LOO equivalence.
Pick one name and use it consistently across chapters. Confirm MRR is reported
@K (it is: MRR@{5,10,20}), and update the MRR definition if the draft implies a
single unbounded MRR.

### C6. Evaluation protocol: the **sampled-candidate** design is undocumented

**Where:** §1.8.2 says only "Evaluation protocols follow those established in prior
multi-behaviour recommendation work" — it never describes the actual protocol.

**Reality (must be written up in Ch. 3):**
- **Leave-one-out** split: each user's last interaction = test, second-last =
  validation, remainder = train (temporal, per-user, no leakage).
- **Sampled candidates:** 1 held-out positive + **10,000 time-valid negatives**
  (seed = 42), from `candidates.npz`; rank = 1 + #negatives scoring above the
  positive.
- This is a **biased** estimator of the full-ranking metric — cite
  **Krichene & Rendle (KDD 2020)** as a limitation. Add this citation.

**Change:** add a methodology subsection describing the protocol, and add the
sampled-eval bias to §1.9 (Limitations).

### C7. Auxiliary behaviours and the missing **negative** behaviour

**Where:** §1.4 Obj. III, §1.8.1, Definition of Terms (**Auxiliary Behaviour**) —
all list auxiliaries as **click, like, comment** and never mention a *negative*
behaviour.

**Reality:** the launcher uses `relation='click,comment,like,gift,negative'`.
With **click as target**, the auxiliaries are **comment, like, gift, negative**.
KuaiLive ships a **`negative.csv`** (~3.9 M short-watch/skip edges) that the draft
never introduces, and the final MBGCN run *includes* it (learned weight ≈ −0.008;
weights: click 1.44, comment 1.60, like 1.69, gift 1.32, negative −0.008 — see
`docs/THESIS_MODIFICATIONS.md`). Note that `docs/MODEL_REVIEW.md` §3 had flagged
`negative` as risky to treat as a positive-style co-occurrence edge; the final
model keeps it but learns it a near-zero/negative weight, which is worth
reporting as a finding.

**Change:** (i) introduce the **negative** behaviour in the dataset description;
(ii) update every auxiliary-behaviour list to match the chosen target (if target =
click, auxiliaries = comment/like/gift/negative); (iii) report the learned
behaviour weights as a result, including the negative-behaviour sign.

### C8. Computational constraints: room subsampling was **not** needed

**Where:** §1.9.5 — implies full-scale graph experiments "may require subsampling
strategies that introduce selection biases."

**Reality:** no room subsampling was applied. The memory problem was solved
differently: (a) items are modelled at **streamer** granularity (452 k nodes, not
11.6 M rooms), and (b) the 43 GB densification bug in `__calculate_user_behaviour`
was replaced with sparse reductions (`docs/MODEL_REVIEW.md` §3,
`docs/THESIS_MODIFICATIONS.md` §1–3). So the cold-start selection-bias worry in
§1.9.5 does not apply.

**Change:** rewrite §1.9.5 to describe the *actual* memory strategy (streamer-level
items + sparse ops) and drop the subsampling-bias caveat, or replace it with the
real limitation (streamer-level aggregation collapses room-level signal).

---

## Priority 3 — Additions needed for Part II (ATRank), not yet in any chapter

These are not corrections to Ch. 1 — they are content that must appear in the
methodology/experiments chapters once ATRank is built.

- **A1. Loss-function harmonisation.** MBGCN uses **BPR** pairwise loss
  (`loss.py`). ATRank's native loss is **point-wise sigmoid cross-entropy** (the
  paper found it beat pairwise). Decide and document: either harmonise both to one
  loss for strict fairness, or keep each model's native loss and argue the
  comparison is of *architectures as intended*. Record the choice.
- **A2. Sequence construction.** ATRank needs a per-user **chronological
  sequence** view of the *same* click interactions (behaviour tuple {action,
  object=streamer, timestamp}, elapsed-time bucketed into exponentially growing
  intervals). This derived view does not exist yet — the `click_interaction`
  dataset only emits graph-shaped `.txt` files. Document the sequence-derivation
  step.
- **A3. Shared evaluation.** ATRank must rank the **identical** `candidates.npz`
  pools with the **identical** metrics, overriding ATRank's native AUC metric.
  Document that the metric was standardised across both models.
- **A4. Heterogeneity mapping.** KuaiLive has one object type (streamer) but many
  action types, so behaviour heterogeneity is carried by ATRank's **action-type
  embedding**, not by object grouping as in the original paper. Document this
  adaptation.

---

## Suggested edit checklist (quick reference)

| # | Section(s) | Change | Type |
|---|-----------|--------|------|
| C1 | 1.4, 1.6, RQ2, RQ4, 1.8.1, Defs | Resolve gift-vs-click target; add "Target Behaviour Selection" | **[YOUR CALL]** |
| C2 | 1.1.4, 1.2, 1.8.1, RQ3, H3, Defs | Item = streamer; ephemerality → candidate eligibility; add "Unit of Recommendation" | **[YOUR CALL]** |
| C3 | 1.9.4 | Rewrite: ID-only primary comparison; side features → ablation only | factual |
| C4 | 1.8.2, Refs | Remove PyTorch Geometric / Fey & Lenssen; state custom sparse PyTorch | factual |
| C5 | 1.8.2, Defs | Reconcile Recall@K vs HR@K naming (LOO-equivalent) | factual |
| C6 | 1.8.2, 1.9, Refs | Document sampled-candidate LOO protocol; add Krichene & Rendle (2020) bias caveat | addition |
| C7 | 1.4, 1.8.1, Defs | Add `negative` behaviour; fix auxiliary list to match target; report learned weights | factual |
| C8 | 1.9.5 | Rewrite memory strategy (streamer items + sparse ops); drop subsampling-bias claim | factual |
| A1–A4 | Ch. 3+ | ATRank: loss choice, sequence build, shared eval, heterogeneity mapping | addition |

**Do C1 and C2 first** — every other section inherits their wording. Nothing below
Priority 1 is worth touching until the target and the item unit are locked.
