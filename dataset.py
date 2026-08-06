import torch
import numpy as np
import os
import random
import scipy.sparse as sp
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
    Here relation include all relations
    """

    def __init__(self, flags_obj):
        self.path = flags_obj.path
        self.name = flags_obj.dataset_name
        self.__decode_relation(flags_obj)
        self.__load_size()
        self.__create_relation_matrix()
        self.__calculate_user_behaviour()
        self.__generate_ground_truth()
        self.__generate_train_matrix()
        # self.__load_item_graph()
        # (user, positive) target pairs; negatives are drawn on the fly.
        self.train_pairs = torch.from_numpy(self.checkins).long()
        self.cnt = 0
        self.newit()

    def __decode_relation(self, flags_obj):
        relation = flags_obj.relation[0]
        self.relation = str(relation).split(",")

    def __load_size(self):
        print(self.path, self.name, "\n\n\n\n")
        with open(os.path.join(self.path, self.name, "data_size.txt")) as f:
            data = f.readline()
            user_num, item_num = data.strip().split()
            self.user_num = int(user_num)
            self.item_num = int(item_num)

    def __load_item_graph(self):
        self.item_graph = {}
        self.item_graph_degree = {}
        for tmp_relation in self.relation:
            self.item_graph[tmp_relation] = torch.load(
                os.path.join(self.path, self.name, "item_" + tmp_relation + ".pth")
            )
            self.item_graph_degree[tmp_relation] = (
                self.item_graph[tmp_relation].sum(dim=1).float().unsqueeze(-1)
            )

    def __create_relation_matrix(self):
        """
        create a matrix for every relation
        """
        self.relation_dict = {}

        for i in range(len(self.relation)):
            index = []
            with open(
                os.path.join(self.path, self.name, self.relation[i] + ".txt")
            ) as f:
                data = f.readlines()
                for row in data:
                    user, item = row.strip().split()
                    user, item = int(user), int(item)
                    index.append([user, item])
            index_tensor = torch.LongTensor(index)
            lens, _ = index_tensor.shape
            self.relation_dict[self.relation[i]] = torch.sparse_coo_tensor(
                index_tensor.t(),
                torch.ones(lens, dtype=torch.float32),
                torch.Size([self.user_num, self.item_num]),
            )

    def __calculate_user_behaviour(self):
        # Degree = interaction count per user/item under each behaviour.
        # Computed with sparse reductions instead of densifying the
        # [user_num x item_num] matrices, which would allocate tens of GB
        # on large datasets (e.g. KuaiLive: 23772 x 452621 ~ 43GB each).
        user_cols = []
        item_cols = []
        for i in range(len(self.relation)):
            mat = self.relation_dict[self.relation[i]]
            # row sums -> per-user interaction count under this behaviour
            user_deg = torch.sparse.sum(mat, dim=1).to_dense().unsqueeze(-1)
            # column sums -> per-item interaction count under this behaviour
            item_deg = torch.sparse.sum(mat, dim=0).to_dense().unsqueeze(-1)
            user_cols.append(user_deg)
            item_cols.append(item_deg)
        self.user_behaviour_degree = torch.cat(user_cols, dim=1)
        self.item_behaviour_degree = torch.cat(item_cols, dim=1)

    def __generate_ground_truth(self):
        """
        use train data to build the ground truth matrix
        """
        row_data = []
        col = []
        with open(os.path.join(self.path, self.name, "train.txt")) as f:
            data = f.readlines()
            for row in data:
                user, item = row.strip().split()
                user, item = int(user), int(item)
                row_data.append(user)
                col.append(item)
        row_data = np.array(row_data)
        col = np.array(col)
        values = np.ones(len(row_data), dtype=float)
        self.ground_truth = sp.csr_matrix(
            (values, (row_data, col)), shape=(self.user_num, self.item_num)
        )
        self.checkins = np.concatenate((row_data[:, None], col[:, None]), axis=1)

    def __generate_train_matrix(self):
        """
        bring all relation together to a big matrix for GCN
        """
        index = []
        with open(os.path.join(self.path, self.name, "train.txt")) as f:
            data = f.readlines()
            for row in data:
                user, item = row.strip().split()
                user, item = int(user), int(item)
                index.append([user, item])
        index_tensor = torch.LongTensor(index)
        lens, _ = index_tensor.shape
        self.train_matrix = torch.sparse_coo_tensor(
            index_tensor.t(),
            torch.ones(lens, dtype=torch.float32),
            torch.Size([self.user_num, self.item_num]),
        )

    def newit(self):
        """Resample one BPR negative per (user, positive) pair for the epoch.

        Replaces the old pre-generated ``sample_file/sample_i.txt`` files, which
        did not scale to the click target (~2.76M pairs x hundreds of epochs).
        Negatives are drawn uniformly over the catalogue; with 452k items and a
        few hundred positives per user the false-negative rate is negligible, so
        no per-user exclusion is done (standard BPR practice).
        """
        n = self.train_pairs.shape[0]
        self.train_neg = torch.randint(0, self.item_num, (n,), dtype=torch.long)
        self.cnt += 1

    def __getitem__(self, index):
        user = self.train_pairs[index, 0].unsqueeze(-1)          # [1]
        items = torch.stack(
            (self.train_pairs[index, 1], self.train_neg[index])  # [pos, neg]
        )
        return user, items

    def __len__(self):
        return len(self.checkins)


class TestDataset(Dataset):
    """Sampled-candidate evaluation (KuaiLive protocol).

    Each instance is one held-out positive plus a fixed pool of pre-sampled,
    time-valid negatives (built by build_candidates.py -> candidates.npz). The
    candidate row returned for a user is ``[positive, neg_0, ..., neg_{N-1}]``
    with the positive at column 0, so scoring + ranking never touches the full
    catalogue and no train mask is needed (negatives already exclude seen items).
    """

    def __init__(self, flags_obj, trainset, task="test"):
        self.path = flags_obj.path
        self.name = flags_obj.dataset_name
        self.task = task
        d = np.load(os.path.join(self.path, self.name, "candidates.npz"))
        prefix = "val" if task == "validation" else "test"
        self.users = d[prefix + "_user"].astype(np.int64)
        self.pos = d[prefix + "_item"].astype(np.int64)
        self.neg = d[prefix + "_neg"]                 # [N, num_neg] int32
        self.num_neg = int(d["num_neg"])
        d.close()

    def __getitem__(self, index):
        cand = np.empty(1 + self.num_neg, dtype=np.int64)
        cand[0] = self.pos[index]                     # positive at column 0
        cand[1:] = self.neg[index]
        return int(self.users[index]), torch.from_numpy(cand)

    def __len__(self):
        return len(self.users)


def test_collate(batch):
    """Collate into (users [B], candidates [B, 1 + num_neg])."""
    users = torch.tensor([b[0] for b in batch], dtype=torch.long)
    candidates = torch.stack([b[1] for b in batch], dim=0)
    return users, candidates
