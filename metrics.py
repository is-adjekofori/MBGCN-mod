import torch
import numpy as np

_is_hit_cache = {}


def get_is_hit(scores, gt, topk):
    '''
    Whether each of the top-k scored items is a ground-truth positive.

    Ground truth is passed sparsely as ``gt = (pos_codes, num_pos, item_num)``
    where ``pos_codes`` are the flattened ``row * item_num + col`` codes of the
    positives for this batch. This avoids materialising a dense
    ``[batch, item_num]`` one-hot just to look items up.

    Returns a ``[batch, topk]`` boolean tensor.
    '''
    global _is_hit_cache
    pos_codes, num_pos, item_num = gt
    cacheid = (id(scores), id(pos_codes))
    if topk in _is_hit_cache and _is_hit_cache[topk]['id'] == cacheid:
        return _is_hit_cache[topk]['is_hit']

    device = scores.device
    batch = scores.shape[0]
    _, col_indice = torch.topk(scores, topk)                       # [batch, topk]
    row_indice = torch.arange(batch, device=device).view(-1, 1).expand(-1, topk)
    codes = row_indice.to(torch.int64) * item_num + col_indice.to(torch.int64)
    is_hit = torch.isin(codes, pos_codes)                          # [batch, topk] bool
    _is_hit_cache[topk] = {'id': cacheid, 'is_hit': is_hit}
    return is_hit


class _Metric:
    '''
    base class of metrics like HR@k NDCG@k
    '''

    def __init__(self):
        self.start()

    @property
    def metric(self):
        return self._metric

    def __call__(self, scores, gt):
        '''
        - scores: model output, shape=(batch, item_num)
        - gt: (pos_codes, num_pos, item_num) sparse ground truth for the batch.
        '''
        raise NotImplementedError

    def get_title(self):
        raise NotImplementedError

    def start(self):
        '''
        clear all
        '''
        global _is_hit_cache
        _is_hit_cache = {}
        self._cnt = 0
        self._metric = 0
        self._sum = 0

    def stop(self):
        global _is_hit_cache
        _is_hit_cache = {}
        self._metric = self._sum/self._cnt


class Recall(_Metric):
    '''
    Recall in top-k samples
    '''

    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.epison = 1e-8

    def get_title(self):
        return "Recall@{}".format(self.topk)

    def __call__(self, scores, gt):
        _, num_pos, _ = gt

        is_hit = get_is_hit(scores, gt, self.topk).sum(dim=1)

        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += (is_hit/(num_pos+self.epison)).sum().item()


class NDCG(_Metric):
    '''
    NDCG in top-k samples
    In this work, NDCG = log(2)/log(1+hit_positions)
    '''

    def DCG(self, hit, device=torch.device('cpu')):

        hit = hit.float()/torch.log2(torch.arange(2, self.topk+2, device=device, dtype=torch.float32))
        return hit.sum(-1)

    def IDCG(self, num_pos):
        hit = torch.zeros(self.topk, dtype=torch.float)
        hit[:num_pos] = 1
        return self.DCG(hit)

    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.IDCGs = torch.empty(1 + self.topk, dtype=torch.float)
        self.IDCGs[0] = 1  # avoid 0/0
        for i in range(1, self.topk + 1):
            self.IDCGs[i] = self.IDCG(i)

    def get_title(self):
        return "NDCG@{}".format(self.topk)

    def __call__(self, scores, gt):
        _, num_pos, _ = gt
        device = scores.device
        is_hit = get_is_hit(scores, gt, self.topk)
        num_pos = num_pos.clamp(0, self.topk).to(torch.long)
        dcg = self.DCG(is_hit, device)
        idcg = self.IDCGs[num_pos.cpu()]
        ndcg = dcg/idcg.to(device)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += ndcg.sum().item()


class MRR(_Metric):
    '''
    Mean reciprocal rank in top-k samples
    '''

    def __init__(self, topk):
        super().__init__()
        self.topk = topk
        self.denominator = torch.arange(1, self.topk+1, dtype=torch.float)

    def get_title(self):
        return "MRR@{}".format(self.topk)

    def __call__(self, scores, gt):
        _, num_pos, _ = gt
        device = scores.device
        is_hit = get_is_hit(scores, gt, self.topk).float()
        is_hit /= self.denominator.to(device)
        first_hit_rr = is_hit.max(dim=1)[0]
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += first_hit_rr.sum().item()


# ---------------------------------------------------------------------------
# Sampled-candidate metrics (KuaiLive protocol)
#
# Each evaluation instance ranks exactly ONE held-out positive against a fixed
# pool of sampled negatives. We pass in the positive's 1-indexed ``rank`` within
# its candidate pool (rank == 1 means it beat every negative). With a single
# positive, Recall@k == HitRate@k and the ideal DCG is 1, so:
#     Recall@k = mean( rank <= k )
#     NDCG@k   = mean( [rank <= k] / log2(rank + 1) )
#     MRR@k    = mean( [rank <= k] / rank )
# ---------------------------------------------------------------------------
class _SampledMetric:
    def __init__(self, topk):
        self.topk = topk
        self.start()

    @property
    def metric(self):
        return self._metric

    def start(self):
        self._cnt = 0
        self._sum = 0.0
        self._metric = 0.0

    def stop(self):
        self._metric = self._sum / max(self._cnt, 1)

    def __call__(self, rank):
        raise NotImplementedError


class SampledRecall(_SampledMetric):
    def get_title(self):
        return "Recall@{}".format(self.topk)

    def __call__(self, rank):
        self._cnt += rank.numel()
        self._sum += (rank <= self.topk).sum().item()


class SampledNDCG(_SampledMetric):
    def get_title(self):
        return "NDCG@{}".format(self.topk)

    def __call__(self, rank):
        hit = (rank <= self.topk).float()
        self._cnt += rank.numel()
        self._sum += (hit / torch.log2(rank.float() + 1.0)).sum().item()


class SampledMRR(_SampledMetric):
    def get_title(self):
        return "MRR@{}".format(self.topk)

    def __call__(self, rank):
        hit = (rank <= self.topk).float()
        self._cnt += rank.numel()
        self._sum += (hit / rank.float()).sum().item()
