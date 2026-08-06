from pickle import TRUE
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from time import time
from tqdm import tqdm
import os
import sys

from loss import bprloss
from dataset import TrainDataset, TestDataset, test_collate
from utils import EarlyStopManager, ModelSelector, VisManager
from metrics import SampledRecall, SampledNDCG, SampledMRR


class TrainManager(object):
    def __init__(self, flags_obj, vm, cm):
        self.flags_obj = flags_obj
        self.vm = vm
        self.cm = cm
        self.es = EarlyStopManager(flags_obj)
        self.data_set_init(flags_obj)
        self.set_device(flags_obj)
        self.model = ModelSelector.getModel(
            flags_obj, self.trainset, self.flags_obj.model, self.device
        ).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=flags_obj.lr)

        # Sampled-candidate metrics at the KuaiLive paper's k = {5, 10, 20}.
        # The first entry (Recall10) drives early stopping / the leaderboard.
        self.metric_dict = {
            "Recall10": SampledRecall(10),
            "NDCG10": SampledNDCG(10),
            "MRR10": SampledMRR(10),
            "Recall20": SampledRecall(20),
            "NDCG20": SampledNDCG(20),
            "MRR20": SampledMRR(20),
            "Recall5": SampledRecall(5),
            "NDCG5": SampledNDCG(5),
            "MRR5": SampledMRR(5),
        }

    def set_device(self, flags_obj):
        if flags_obj.gpu == True:
            torch.cuda.set_device(flags_obj.gpu_id)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def data_set_init(self, flags_obj):
        self.trainset = TrainDataset(flags_obj)
        print("Train Data Read Completed!")
        self.validationset = TestDataset(flags_obj, self.trainset, task="validation")
        print("Validation Data Read Completed!")
        self.testset = TestDataset(flags_obj, self.trainset, task="test")
        print("Test Data Read Completed!")
        self.trainloader = DataLoader(
            self.trainset,
            flags_obj.batch_size,
            True,
            num_workers=flags_obj.num_workers,
            pin_memory=True,
        )
        self.validationloader = DataLoader(
            self.validationset,
            flags_obj.test_batch_size,
            False,
            num_workers=flags_obj.num_workers,
            pin_memory=True,
            collate_fn=test_collate,
        )
        self.testloader = DataLoader(
            self.testset,
            flags_obj.test_batch_size,
            False,
            num_workers=flags_obj.num_workers,
            pin_memory=True,
            collate_fn=test_collate,
        )

    def train(self):
        self.set_leaderboard()

        for epoch in range(self.flags_obj.epoch):
            self.train_one_epoch()

            is_last = (epoch + 1) == self.flags_obj.epoch
            if (epoch + 1) % self.flags_obj.eval_every == 0 or is_last:
                self.validation()
                self.update_leaderboard(epoch)
                stop = self.es.step(
                    list(self.metric_dict.values())[0]._metric, epoch
                )
                if stop == True:
                    break

            self.trainloader.dataset.newit()

    def train_one_epoch(self):
        self.model.train()
        if self.flags_obj.model == "MBGCN":
            print(self.model.mgnn_weight)

        start = time()
        total_loss = 0
        for i, data in enumerate(tqdm(self.trainloader)):
            users, items = data
            self.opt.zero_grad()
            modelout = self.model(users.to(self.device), items.to(self.device))

            loss = bprloss(
                modelout,
                batch_size=self.trainloader.batch_size,
                loss_mode=self.flags_obj.loss_mode,
            )
            total_loss += loss

            loss.backward()
            self.opt.step()

        time_interval = time() - start

        self.vm.update_line("epoch loss", total_loss)
        self.vm.update_line("train time cost", time_interval)

    def run_metrics(self, loader):
        """Sampled-candidate evaluation over `loader`.

        Propagation is done once; each batch scores only its users against their
        pre-sampled candidate pool ([positive, negatives...] with the positive at
        column 0). The positive's rank within the pool is
        ``1 + #negatives scoring higher``; all metrics are derived from that rank.
        """
        self.model.eval()
        for metric in self.metric_dict:
            self.metric_dict[metric].start()
        with torch.no_grad():
            propagate_result = self.model.propagate(task="test")
            for users, candidates in tqdm(loader):
                scores = self.model.evaluate_candidates(
                    propagate_result,
                    users.to(self.device),
                    candidates.to(self.device),
                )                                          # [B, 1 + num_neg]
                pos_score = scores[:, 0:1]
                rank = (scores[:, 1:] > pos_score).sum(dim=1) + 1   # [B]

                for metric in self.metric_dict:
                    self.metric_dict[metric](rank)

        for metric in self.metric_dict:
            self.metric_dict[metric].stop()

    def validation(self):
        start = time()
        self.run_metrics(self.validationloader)
        time_interval = time() - start

        self.vm.update_line("validation time cost", time_interval)
        self.vm.update_metrics(self.metric_dict)

        for metric in self.metric_dict:
            print("{}:{}".format(metric, self.metric_dict[metric]._metric))

    def set_leaderboard(self):
        self.max_metric = -1.0
        self.max_epoch = -1
        self.leaderboard = self.vm.new_text_window("leaderboard")

    def update_leaderboard(self, epoch):

        metric_list = list(self.metric_dict.values())
        metric = metric_list[0]._metric
        if metric > self.max_metric:
            print("Here\n\n\n\n")
            self.max_metric = metric
            self.max_epoch = epoch

            self.vm.append_text(
                "New Record! {} @ epoch {}!".format(metric, epoch), self.leaderboard
            )
            self.cm.model_save(self.model)
            print(metric, self.max_metric)

    def test(self):
        # Evaluate the best saved model on the held-out test set.
        self.cm.model_load(self.model)
        self.run_metrics(self.testloader)

        self.vm.append_text("Final Test Result:", self.leaderboard)
        for metric in self.metric_dict:
            self.vm.append_text(
                metric + ": {}".format(self.metric_dict[metric]._metric)
            )
