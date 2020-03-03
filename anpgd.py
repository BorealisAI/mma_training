# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from advertorch.attacks import Attack
from advertorch.attacks import LabelMixin
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.utils import batch_multiply
from advertorch.utils import clamp
from advertorch.loss import elementwise_margin


def bisection_search(
        cur_eps, ptb, model, data, label, fn_margin, margin_init,
        maxeps, num_steps,
        cur_min=None, clip_min=0., clip_max=1.):

    assert torch.all(cur_eps <= maxeps)

    margin = margin_init

    if cur_min is None:
        cur_min = torch.zeros_like(margin)
    cur_max = maxeps.clone().detach()

    for ii in range(num_steps):
        cur_min = torch.max((margin < 0).float() * cur_eps, cur_min)
        cur_max = torch.min(((margin < 0).float() * maxeps
                             + (margin >= 0).float() * cur_eps),
                            cur_max)

        cur_eps = (cur_min + cur_max) / 2
        margin = fn_margin(
            model(clamp(data + batch_multiply(cur_eps, ptb),
                        min=clip_min, max=clip_max)
                  ), label)

    assert torch.all(cur_eps <= maxeps)

    return cur_eps


class ANPGD(Attack, LabelMixin):

    def __init__(self, pgdadv, mineps, maxeps, num_search_steps,
                 eps_iter_scale, search_loss_fn=None):
        self.pgdadv = pgdadv
        self.predict = self.pgdadv.predict
        self.mineps = mineps  # mineps is used outside to set prev_eps
        self.maxeps = maxeps
        self.num_search_steps = num_search_steps
        self.eps_iter_scale = eps_iter_scale
        assert search_loss_fn is not None
        self.search_loss_fn = search_loss_fn


    def _get_unitptb_and_eps(self, xadv, x, y, prev_eps):
        unitptb = batch_multiply(1. / (prev_eps + 1e-12), (xadv - x))
        logit_margin = self.search_loss_fn(self.predict(xadv), y)

        maxeps = self.maxeps * torch.ones_like(y).float()

        curr_eps = bisection_search(
            prev_eps, unitptb, self.predict, x, y, self.search_loss_fn,
            logit_margin, maxeps, self.num_search_steps)
        return unitptb, curr_eps

    def perturb(self, x, y, prev_eps):

        self.pgdadv.eps = prev_eps
        self.pgdadv.eps_iter = self.scale_eps_iter(
            self.pgdadv.eps, self.pgdadv.nb_iter)
        with ctx_noparamgrad_and_eval(self.predict):
            xadv = self.pgdadv.perturb(x, y)

        unitptb, curr_eps = self._get_unitptb_and_eps(xadv, x, y, prev_eps)

        xadv = x + batch_multiply(curr_eps, unitptb)
        return xadv, curr_eps

    def scale_eps_iter(self, eps, nb_iter):
        return self.eps_iter_scale * eps / nb_iter



class ANPGDForTest(Attack, LabelMixin):
    # XXX: consider merge ANPGDForTest and ANPGD together later

    def __init__(self, pgdadv, maxeps, num_search_steps):
        self.pgdadv = pgdadv
        self.predict = self.pgdadv.predict
        self.maxeps = maxeps
        self.num_search_steps = num_search_steps

    # XXX: largely duplicate
    def _get_unitptb_and_eps(self, xadv, x, y, prev_eps):
        unitptb = batch_multiply(1. / (prev_eps + 1e-12), (xadv - x))
        logit_margin = elementwise_margin(self.predict(xadv), y)

        ones = torch.ones_like(y).float()
        maxeps = self.maxeps * ones

        curr_eps = bisection_search(
            maxeps * 0.5, unitptb, self.predict, x, y, elementwise_margin,
            logit_margin, maxeps, self.num_search_steps)
        return unitptb, curr_eps

    def perturb(self, x, y):
        with ctx_noparamgrad_and_eval(self.predict):
            xadv = self.pgdadv.perturb(x, y)

        unitptb, curr_eps = self._get_unitptb_and_eps(
            xadv, x, y, self.pgdadv.eps)
        return xadv, curr_eps
