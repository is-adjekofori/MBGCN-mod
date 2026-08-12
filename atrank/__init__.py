"""ATRank (Zhou et al., AAAI 2018) reimplemented in PyTorch for KuaiLive.

Our own implementation — NOT a fork of reference/atrank_upstream/ (that is the
authors' TensorFlow code, kept read-only for reference). This package obeys the
fair-comparison contract with MBGCN: identical click target, identical
leave-one-out split, identical candidates.npz pools, identical Recall/NDCG/MRR@k.

Object embeddings are ID-only (streamer id + action-type + elapsed-time bucket);
side features are deliberately excluded from the primary comparison.
"""
