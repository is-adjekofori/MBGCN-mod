import torch
import numpy as np
import pandas as pd

import csv
import random
import setproctitle
import os
import sys

from absl import logging
from absl import flags

from model import MF, MBGCN
from dataset import TrainDataset

# NOTE: visdom is imported lazily inside VisManager.set_visdom so that headless
# runs (e.g. Colab, --no_vis) neither require the package nor a running server.


def csv_to_tensor(filepath, columns, dtype=torch.float32):
    """
    columns: str (single column) or list[str] (multiple columns)
    """
    df = pd.read_csv(filepath)
    data = df[columns].values
    return torch.tensor(data, dtype=dtype)


class ContentManager(object):
    def __init__(self, flag_obj):
        self.name = flag_obj.name
        self.dataset_name = flag_obj.dataset_name
        self.output_path = flag_obj.output
        self.path = os.path.join(self.output_path, self.dataset_name, self.name)
        self.set_proctitle()
        self.output_init()

    def set_proctitle(self):
        setproctitle.setproctitle(self.name)

    def output_init(self):
        if not os.path.exists(os.path.join(self.output_path, self.dataset_name)):
            os.mkdir(os.path.join(self.output_path, self.dataset_name))

        if not os.path.exists(
            os.path.join(self.output_path, self.dataset_name, self.name)
        ):
            os.mkdir(os.path.join(self.output_path, self.dataset_name, self.name))

    def model_save(self, model):
        torch.save(model.state_dict(), os.path.join(self.path, "model.pkl"))

    def model_load(self, model):
        model.load_state_dict(torch.load(os.path.join(self.path, "model.pkl")))

    # --- full training checkpoint (for resume) -----------------------------
    # model.pkl holds only the best weights (for final eval); checkpoint.pkl
    # additionally holds optimizer + epoch + early-stop/leaderboard state so a
    # resumed run continues faithfully rather than restarting from scratch.
    @property
    def checkpoint_path(self):
        return os.path.join(self.path, "checkpoint.pkl")

    def has_checkpoint(self):
        return os.path.exists(self.checkpoint_path)

    def save_checkpoint(self, state):
        torch.save(state, self.checkpoint_path)

    def load_checkpoint(self, map_location="cpu"):
        return torch.load(self.checkpoint_path, map_location=map_location)


class VisManager(object):
    def __init__(self, flag_obj):
        self.name = flag_obj.name + "_vm"
        self.exp_name = flag_obj.name
        self.__get_port(flag_obj)
        # Headless when --no_vis is set (Colab / no visdom server). All plotting
        # calls degrade to stdout prints; the training/eval loop is unaffected.
        self.headless = bool(getattr(flag_obj, "no_vis", False))
        self.__init_logging(flag_obj)
        self.set_visdom()
        self.show_basic_info(flag_obj)

    def __get_port(self, flag_obj):
        self.port = flag_obj.port

    def __init_logging(self, flag_obj):
        """Set up durable CSV logs, independent of visdom.

        Written under the same dir as the model checkpoint:
          <output>/<dataset>/<name>/metrics.csv  -- one row per validation
                                                     (epoch + every metric), and
          <output>/<dataset>/<name>/scalars.csv  -- long format (epoch,name,value)
                                                     for loss / timing / metrics.
        Use these to plot learning curves for the thesis.
        """
        self.log_dir = os.path.join(
            flag_obj.output, flag_obj.dataset_name, flag_obj.name
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_csv = os.path.join(self.log_dir, "metrics.csv")
        self.scalars_csv = os.path.join(self.log_dir, "scalars.csv")
        self.current_epoch = -1

    def set_epoch(self, epoch):
        """Tell the logger which epoch subsequent updates belong to."""
        self.current_epoch = epoch

    def resume_from(self, start_epoch):
        """Drop already-logged rows at ``epoch >= start_epoch`` before resuming.

        Keeps the logs single-valued per epoch when a run is restarted from a
        checkpoint (the epochs from start_epoch onward will be re-logged).
        """
        for path in (self.metrics_csv, self.scalars_csv):
            self.__prune_epochs(path, start_epoch)

    def __prune_epochs(self, path, start_epoch):
        if not os.path.exists(path):
            return
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return
        header, data = rows[0], rows[1:]
        ei = header.index("epoch")
        kept = [r for r in data if r and int(float(r[ei])) < start_epoch]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(kept)

    def _append_row(self, path, header, row):
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)

    def set_visdom(self):
        if self.headless:
            self.vis = None
            return
        from visdom import Visdom  # lazy: only needed for the live dashboard
        self.vis = Visdom(port=self.port, env=self.exp_name)

    def show_basic_info(self, flag_obj):
        info = [
            ("Name", flag_obj.name),
            ("Model", flag_obj.model),
            ("Dataset", flag_obj.dataset_name),
            ("Embedding Size", flag_obj.embedding_size),
            ("Node dropout", flag_obj.node_dropout),
            ("Message dropout", flag_obj.message_dropout),
            ("Initial lr", flag_obj.lr),
            ("Batch size", flag_obj.batch_size),
            ("Early stop patience", flag_obj.es_patience),
            ("L2_norm", flag_obj.L2_norm),
        ]
        if self.vis is None:
            print("=== Basic Information ===")
            for k, v in info:
                print("  {}: {}".format(k, v))
            self.basic = None
            return

        basic = self.vis.text("Basic Information:")
        for k, v in info:
            self.vis.text("{}: {}".format(k, v), win=basic, append=True)
        self.basic = basic

    def update_line(self, title, value):
        if type(value) == torch.Tensor:
            value = value.item()

        # durable log (both headless and visdom modes)
        self._append_row(
            self.scalars_csv, ["epoch", "name", "value"],
            [self.current_epoch, title, value],
        )

        if self.vis is None:
            print("[{}] {}".format(title, value))
            return

        if not hasattr(self, title):
            setattr(self, title, self.vis.line([value], [0], opts=dict(title=title)))
            setattr(self, title + "_step", 1)
        else:
            step = getattr(self, title + "_step")
            self.vis.line([value], [step], win=getattr(self, title), update="append")
            setattr(self, title + "_step", step + 1)

    def update_metrics(self, record):
        # one wide row per validation: epoch + every metric (easy to chart)
        titles = list(record.keys())
        self._append_row(
            self.metrics_csv,
            ["epoch"] + titles,
            [self.current_epoch] + [record[t]._metric for t in titles],
        )
        for title, value in record.items():
            self.update_line(title, value._metric)

    def new_text_window(self, title):
        if self.vis is None:
            print("=== {} ===".format(title))
            return title
        return self.vis.text(title)

    def append_text(self, text, win=None):
        if self.vis is None:
            print(text)
            return
        self.vis.text(text, win=win, append=True)


class EarlyStopManager(object):
    def __init__(self, flags_obj):
        self.es_patience = flags_obj.es_patience
        self.count = 0
        self.max_metric = 0

    def step(self, metric, epoch):
        if epoch <= 10:
            return False

        if metric > self.max_metric:
            self.max_metric = metric
            self.count = 0
            return False
        else:
            self.count += 1
            if self.count < self.es_patience:
                return False
            else:
                return True


class ModelSelector(object):
    def __init__(self):
        pass

    @staticmethod
    def getModel(flags_obj, dataset, model_name, device):
        if model_name == "MF":
            return MF(flags_obj, dataset, device)
        elif model_name == "MBGCN":
            return MBGCN(flags_obj, dataset, device)
        else:
            raise ValueError("Model name is not correct!")
