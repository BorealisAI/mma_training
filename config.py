# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy


class Config(object):
    pass


BaseConfig = Config()
BaseConfig.clip_min = 0.
BaseConfig.clip_max = 1.
BaseConfig.rand_init = True
BaseConfig.clean_loss_fn = "xent"
BaseConfig.margin_loss_fn = "xent"
BaseConfig.attack_loss_fn = "slm"
BaseConfig.search_loss_fn = "slm"
BaseConfig.clean_loss_coeff = 1. / 3
BaseConfig.eps_iter_scale = 2.5
BaseConfig.num_search_steps = 10

CIFAR10 = copy.deepcopy(BaseConfig)
CIFAR10.nb_iter = 10
CIFAR10.optimizer = "SGD"
CIFAR10.training_batch_size = 128
CIFAR10.max_num_training_steps = 50000
CIFAR10.weight_decay = 0.0002
CIFAR10.momentum = 0.9
CIFAR10.learning_rate_schedule = dict(
    [[0, 0.3], [20000, 0.09], [30000, 0.03], [40000, 0.009]])

LinfCIFAR10 = copy.deepcopy(CIFAR10)
LinfCIFAR10.test_eps = 8. / 255
LinfCIFAR10.test_eps_iter = 2. / 255
LinfCIFAR10.attack_mineps = 0.005

L2CIFAR10 = copy.deepcopy(CIFAR10)
L2CIFAR10.test_eps = 1.
L2CIFAR10.test_eps_iter = 0.25
L2CIFAR10.attack_mineps = 0.5

MNIST = copy.deepcopy(BaseConfig)
MNIST.nb_iter = 40
MNIST.optimizer = "Adam"
MNIST.training_batch_size = 50
MNIST.max_num_training_steps = 100000
MNIST.initial_learning_rate = 1e-4
MNIST.learning_rate_schedule = None

LinfMNIST = copy.deepcopy(MNIST)
LinfMNIST.test_eps = 0.3
LinfMNIST.test_eps_iter = 0.01
LinfMNIST.attack_mineps = 0.1

L2MNIST = copy.deepcopy(MNIST)
L2MNIST.test_eps = 2.
L2MNIST.test_eps_iter = 0.125
L2MNIST.attack_mineps = 0.5

config = {
    "LinfCIFAR10": LinfCIFAR10,
    "LinfMNIST": LinfMNIST,
    "L2MNIST": L2MNIST,
    "L2CIFAR10": L2CIFAR10,
}
