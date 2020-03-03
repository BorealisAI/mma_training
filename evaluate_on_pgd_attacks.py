# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import torch
import torch.nn as nn

from advertorch.attacks import LinfPGDAttack, L2PGDAttack, ChooseBestAttack
from advertorch.attacks.utils import attack_whole_dataset
from advertorch.loss import CWLoss
from advertorch.utils import get_accuracy

from advertorch_examples.models import LeNet5Madry
from advertorch_examples.models import get_cifar10_wrn28_widen_factor
from advertorch_examples.utils import get_test_loader


def generate_adversaries(attack_class, nb_restart, **kwargs):
    nb_cw = nb_restart // 2
    nb_ce = nb_restart - nb_cw
    adversaries = []

    kwargs["loss_fn"] = nn.CrossEntropyLoss(reduction="sum")
    ce_pgd = attack_class(**kwargs)

    adversaries = [ce_pgd] * nb_ce

    if nb_cw > 0:
        kwargs["loss_fn"] = CWLoss(reduction="sum")
        cw_pgd = attack_class(**kwargs)
        adversaries += [cw_pgd] * nb_cw

    return adversaries


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default="cuda")
    parser.add_argument('--deterministic', default=False, action="store_true")
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--norm', required=True, choices=("Linf", "L2"))
    parser.add_argument('--eps', required=True, type=float)
    parser.add_argument('--eps_iter', default=None, type=float)
    parser.add_argument('--nb_iter', default=100, type=int)
    parser.add_argument('--nb_restart', default=10, type=int)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--test_size', default=None, type=int)


    args = parser.parse_args()

    ckpt = torch.load(args.model)


    if args.dataset.upper() == "CIFAR10":
        model = get_cifar10_wrn28_widen_factor(4)
    elif args.dataset.upper() == "MNIST":
        model = LeNet5Madry()
    else:
        raise

    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    model.eval()

    print("model loaded")

    if args.dataset.upper() == "CIFAR10" and args.norm == "Linf" \
            and args.eps > 1.:
        args.eps = round(args.eps / 255., 4)


    if args.eps_iter is None:
        if args.dataset.upper() == "MNIST" and args.norm == "Linf":
            args.eps_iter = args.eps / 40.
        else:
            args.eps_iter = args.eps / 4.

    test_loader = get_test_loader(
        args.dataset.upper(), test_size=args.test_size, batch_size=100)


    if args.norm == "Linf":
        attack_class = LinfPGDAttack
    elif args.norm == "L2":
        attack_class = L2PGDAttack
    else:
        raise


    base_adversaries = generate_adversaries(
        attack_class, args.nb_restart, predict=model, eps=args.eps,
        nb_iter=args.nb_iter, eps_iter=args.eps_iter, rand_init=True)

    adversary = ChooseBestAttack(model, base_adversaries)


    adv, label, pred, advpred = attack_whole_dataset(
        adversary, test_loader, device=args.device)

    print(get_accuracy(advpred, label))
    print(get_accuracy(advpred, pred))

    torch.save({"adv": adv}, os.path.join(
        os.path.dirname(args.model), "advdata_eps-{}.pt".format(args.eps)))
    torch.save(
        {"label": label, "pred": pred, "advpred": advpred},
        os.path.join(os.path.dirname(args.model),
                     "advlabel_eps-{}.pt".format(args.eps)))
