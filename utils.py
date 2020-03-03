# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.utils.data.dataset import Subset


DATA_PATH = "./raw_data"


def add_indexes_to_loader(loader):
    dataset = loader.dataset

    while isinstance(dataset, Subset):  # XXX: there might be multiple layers
        dataset = dataset.dataset

    if dataset.train:
        # XXX: if statements for old torchvision
        if "train_labels" in dataset.__dict__:
            targets = dataset.train_labels
            dataset.train_labels = torch.arange(len(targets))
        else:
            targets = dataset.targets
            dataset.targets = torch.arange(len(targets))
    else:
        # XXX: if statements for old torchvision
        if "test_labels" in dataset.__dict__:
            targets = dataset.test_labels
            dataset.test_labels = torch.arange(len(targets))
        else:
            targets = dataset.targets
            dataset.targets = torch.arange(len(targets))

    loader.targets = torch.tensor(targets)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    # copied from https://github.com/pytorch/examples

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_loss_fn(name, reduction):
    if name == "xent":
        loss_fn = nn.CrossEntropyLoss(reduction=reduction)
    elif name == "slm":
        from advertorch.loss import SoftLogitMarginLoss
        loss_fn = SoftLogitMarginLoss(reduction=reduction)
    elif name == "lm":
        from advertorch.loss import LogitMarginLoss
        loss_fn = LogitMarginLoss(reduction=reduction)
    elif name == "cw":
        from advertorch.loss import CWLoss
        loss_fn = CWLoss(reduction=reduction)
    else:
        raise NotImplementedError("loss_fn={}".format(name))

    return loss_fn


def get_sum_loss_fn(name):
    return get_loss_fn(name, "sum")


def get_mean_loss_fn(name):
    return get_loss_fn(name, "elementwise_mean")


def get_none_loss_fn(name):
    return get_loss_fn(name, "none")
