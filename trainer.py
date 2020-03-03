# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

from collections import OrderedDict
import time
import copy

import numpy as np
import torch
import torch.optim as optim

from advertorch.context import ctx_noparamgrad_and_eval, ctx_eval
from advertorch.utils import predict_from_logits, get_accuracy

from utils import AverageMeter


def init_loss_acc_meter():
    meter = {}
    meter["epoch_loss"] = AverageMeter()
    meter["epoch_acc"] = AverageMeter()
    meter["disp_loss"] = AverageMeter()
    meter["disp_acc"] = AverageMeter()
    return meter


def init_eps_meter():
    meter = {}
    meter["epoch_eps"] = AverageMeter()
    meter["disp_eps"] = AverageMeter()
    return meter


def reset_epoch_loss_acc_meter(meter):
    meter["epoch_loss"] = AverageMeter()
    meter["epoch_acc"] = AverageMeter()


def reset_disp_loss_acc_meter(meter):
    meter["disp_loss"] = AverageMeter()
    meter["disp_acc"] = AverageMeter()


def update_loss_acc_meter(meter, loss, acc, num):
    meter["epoch_loss"].update(loss, num)
    meter["epoch_acc"].update(acc, num)
    meter["disp_loss"].update(loss, num)
    meter["disp_acc"].update(acc, num)


def update_eps_meter(meter, eps, num):
    meter["epoch_eps"].update(eps, num)
    meter["disp_eps"].update(eps, num)


class MetersMixin(object):

    def print_disp_meters(self, batch_idx=None):
        if not self.verbose:
            return

        if batch_idx is not None:
            disp_str = "Epoch: {} ({:.0f}%)".format(
                self.epochs,
                100. * (batch_idx + 1) / len(self.loader),
            )
        else:
            disp_str = ""
        for key in self.meters:
            meter = self.meters[key]
            if key == "eps":
                disp_str += "\tavgeps: {:.4f}".format(meter["disp_eps"].avg)
            elif key in ["cln", "adv", "mix"]:
                disp_str += "\t{}loss: {:.4f}, {}acc: {:.2f}%".format(
                    key, meter["disp_loss"].avg, key,
                    100 * meter["disp_acc"].avg)
            else:
                raise ValueError("key=".format(key))

        print(disp_str)

    def reset_epoch_meters(self):
        for key in self.meters:
            reset_epoch_loss_acc_meter(self.meters[key])

    def reset_disp_meters(self):
        for key in self.meters:
            reset_disp_loss_acc_meter(self.meters[key])

    def init_meters(self):
        self.meters = OrderedDict()
        self.cln_meter = init_loss_acc_meter()
        self.meters["cln"] = self.cln_meter
        self.eps_meter = init_eps_meter()
        self.meters["eps"] = self.eps_meter

    def predict_then_update_loss_acc_meter(self, meter, data, target):
        with torch.no_grad(), ctx_eval(self.model):
            output = self.model(data)
        acc = get_accuracy(predict_from_logits(output), target)
        loss = self.loss_fn(output, target).item()
        update_loss_acc_meter(meter, loss, acc, len(data))
        return loss, acc


class GradCloner(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.clone_model = copy.deepcopy(model)
        self.clone_optimizer = optim.SGD(self.clone_model.parameters(), lr=0.)

    def copy_and_clear_grad(self):
        self.clone_optimizer.zero_grad()
        for (pname, pvalue), (cname, cvalue) in zip(
                self.model.named_parameters(),
                self.clone_model.named_parameters()):
            cvalue.grad = pvalue.grad.clone()
        self.optimizer.zero_grad()

    def combine_grad(self, alpha=1, beta=1):
        for (pname, pvalue), (cname, cvalue) in zip(
                self.model.named_parameters(),
                self.clone_model.named_parameters()):
            pvalue.grad.data = \
                alpha * pvalue.grad.data + beta * cvalue.grad.data


class TrainEvalMixin(object):
    def __init__(self, model, device, loss_fn, loader,
                 dataname, adversary, verbose):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.loader = loader
        self.adversary = adversary
        self.verbose = verbose
        self.dataname = dataname
        self.epochs = 0
        self.init_meters()

        self.dct_eps = {}
        self.dct_eps_record = {}

        self.loader.targets = self.loader.targets.to(self.device)
        self.model.to(self.device)

    def disp_eps_hist(self, bins=10):
        interval = self.adversary.maxeps / bins
        hist_str = []
        hist = []
        thresholds = []
        for ii in range(bins):
            thresholds.append((ii * interval, (ii + 1) * interval))
            hist_str.append("{:.2f} to {:.2f}:".format(
                thresholds[-1][0], thresholds[-1][1]))
            hist.append(0)
        for key in self.dct_eps:
            assigned = False
            for ii in range(bins):
                if thresholds[ii][0] <= self.dct_eps[key] < thresholds[ii][1]:
                    hist[ii] += 1
                    assigned = True
                    break
            if not assigned and np.allclose(
                    self.dct_eps[key], self.adversary.maxeps):
                hist[-1] += 1
                assigned = True
            if not assigned:
                raise ValueError(
                    "Should not reach here, eps={}, maxeps={}".format(
                        self.dct_eps[key], self.adversary.maxeps))
        for hstr, h in zip(hist_str, hist):
            print(hstr, h)

    def update_eps(self, eps, idx):
        for jj, ii in enumerate(idx):
            ii = ii.item()
            curr_epsval = eps[jj].item()
            if ii not in self.dct_eps_record:
                self.dct_eps_record[ii] = []
            self.dct_eps_record[ii].append(
                (curr_epsval, self.epochs, self.steps))
            self.dct_eps[ii] = curr_epsval

    def get_eps(self, idx, data):
        lst_eps = []
        for ii in idx:
            ii = ii.item()
            lst_eps.append(max(
                self.adversary.mineps,
                self.dct_eps.setdefault(ii, self.adversary.mineps)
            ))
        return data.new_tensor(lst_eps)


class Trainer(MetersMixin, TrainEvalMixin):
    def __init__(self, model, device, loss_fn, optimizer, loader,
                 margin_loss_fn, hinge_maxeps, clean_loss_coeff=1. / 3,
                 disp_interval=100, adversary=None,
                 max_steps=None, verbose=True,
                 lr_by_steps=None, lr_by_epochs=None,
                 dataname="train"):
        # lr_by_steps: dict with steps as key, and lr as value
        # lr_by_epochs: dict with epochs as key, and lr as value
        TrainEvalMixin.__init__(
            self, model, device, loss_fn, loader, dataname,
            adversary, verbose)

        self.optimizer = optimizer
        self.hinge_maxeps = hinge_maxeps
        self.margin_loss_fn = margin_loss_fn
        self.clean_loss_coeff = clean_loss_coeff
        self.add_clean_loss = clean_loss_coeff > 0
        self.steps = 0
        self.disp_interval = disp_interval
        self.max_steps = max_steps
        self.keep_training = True

        if (lr_by_epochs is not None) and (lr_by_steps is not None):
            raise ValueError(
                "Only one of lr_by_epochs and lr_by_steps can be not None!")
        self.lr_by_steps = lr_by_steps
        self.lr_by_epochs = lr_by_epochs

        if self.add_clean_loss:
            self.grad_cloner = GradCloner(self.model, self.optimizer)

        self._adjust_lr_by_epochs()
        self._adjust_lr_by_steps()


    def train_one_epoch(self):
        _bgn_epoch = time.time()
        if self.verbose:
            print("Training epoch {}".format(self.epochs))
        self.model.train()
        self.model.to(self.device)
        self.reset_epoch_meters()
        self.reset_disp_meters()

        _train_time = 0.

        for batch_idx, (data, idx) in enumerate(self.loader):
            data, idx = data.to(self.device), idx.to(self.device)
            target = self.loader.targets[idx]

            _bgn_train = time.time()
            clnoutput, clnloss, eps = self.train_one_batch(data, idx, target)
            _train_time = _train_time + (time.time() - _bgn_train)

            clnacc = get_accuracy(predict_from_logits(clnoutput), target)
            update_loss_acc_meter(
                self.cln_meter, clnloss.item(), clnacc, len(data))
            update_eps_meter(self.eps_meter, eps.mean().item(), len(data))

            if self.disp_interval is not None and \
                    batch_idx % self.disp_interval == 0:
                self.print_disp_meters(batch_idx)
                self.reset_disp_meters()

            if self.steps == self.max_steps:
                self.stop_training()
                break

        self.print_disp_meters(batch_idx)
        self.disp_eps_hist()
        self.epochs += 1
        self._adjust_lr_by_epochs()

        print("total epoch time", time.time() - _bgn_epoch)
        print("training total time", _train_time)

    def train_one_batch(self, data, idx, target):
        # clean prediction and save clean gradient
        clnoutput = self.model(data)
        clnloss = self.loss_fn(clnoutput, target)

        if self.add_clean_loss:
            self.optimizer.zero_grad()
            clnloss.backward()
            self.grad_cloner.copy_and_clear_grad()


        # anpgd on correct examples
        search_loss = self.adversary.search_loss_fn(clnoutput, target)
        cln_correct = (search_loss < 0)
        cln_wrong = (search_loss >= 0)

        data_correct = data[cln_correct]
        target_correct = target[cln_correct]
        idx_correct = idx[cln_correct]

        num_correct = cln_correct.sum().item()
        num_wrong = cln_wrong.sum().item()

        curr_eps = data.new_zeros(len(data))
        if num_correct > 0:
            prev_eps = self.get_eps(idx_correct, data)

            advdata_correct, curr_eps_correct = self.adversary(
                data_correct, target_correct, prev_eps)

            data[cln_correct] = advdata_correct
            curr_eps[cln_correct] = curr_eps_correct


        # mma loss and gradient
        mmaoutput = self.model(data)
        if num_correct == 0:
            marginloss = mmaoutput.new_zeros(size=(1,))
        else:
            marginloss = self.margin_loss_fn(
                mmaoutput[cln_correct], target[cln_correct])
        if num_wrong == 0:
            clsloss = 0.
        else:
            clsloss = self.loss_fn(mmaoutput[cln_wrong], target[cln_wrong])

        if num_correct > 0:
            marginloss = marginloss[self.hinge_maxeps > curr_eps_correct]

        mmaloss = (marginloss.sum() + clsloss * num_wrong) / len(data)
        self.optimizer.zero_grad()
        mmaloss.backward()


        # combine gradient from both clean loss and mma loss
        if self.add_clean_loss:
            self.grad_cloner.combine_grad(
                1 - self.clean_loss_coeff, self.clean_loss_coeff)

        self.optimizer.step()

        self.update_eps(curr_eps, idx)
        self.steps += 1
        self._adjust_lr_by_steps()
        return clnoutput, clnloss, curr_eps

    def adjust_lr(self, dct, key):
        # fixed learning rate for all the params
        if dct is not None and key in dct:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = dct[key]
            print("Learning rate adjusted to {}".format(dct[key]))

    def _adjust_lr_by_steps(self):
        self.adjust_lr(self.lr_by_steps, self.steps)

    def _adjust_lr_by_epochs(self):
        self.adjust_lr(self.lr_by_epochs, self.epochs)

    def stop_training(self):
        self.keep_training = False


class Evaluator(MetersMixin, TrainEvalMixin):
    def __init__(self, model, device, loss_fn, loader,
                 dataname="test", adversary=None,
                 verbose=True):
        TrainEvalMixin.__init__(
            self, model, device, loss_fn, loader, dataname,
            adversary, verbose)

        self.adv_meter = init_loss_acc_meter()
        self.meters["adv"] = self.adv_meter
        self.steps = None

    def test_one_epoch(self):
        print("Evaluating on {}, epoch {}".format(self.dataname, self.epochs))
        self.model.eval()
        self.model.to(self.device)
        self.reset_epoch_meters()
        self.reset_disp_meters()

        for data, idx in self.loader:
            data, idx = data.to(self.device), idx.to(self.device)
            target = self.loader.targets[idx]

            with ctx_noparamgrad_and_eval(self.model):
                # this advdata is a fixed eps adv, not scaled
                advdata, curr_eps = self.adversary.perturb(data, target)

            update_eps_meter(self.eps_meter, curr_eps.mean().item(), len(data))
            self.update_eps(curr_eps, idx)
            clnloss, clnacc = self.predict_then_update_loss_acc_meter(
                self.cln_meter, data, target)
            advloss, advacc = self.predict_then_update_loss_acc_meter(
                self.adv_meter, advdata, target)


        self.epochs += 1
        self.print_disp_meters()
        self.disp_eps_hist()

        return (self.meters['cln']['epoch_acc'].avg,
                self.meters['adv']['epoch_acc'].avg,
                np.array(list(self.dct_eps.values())).mean())
