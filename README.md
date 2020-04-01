# MMA Training (Max-Margin Adversarial Training)

This repo contains code for [MMA Training: Direct Input Space Margin Maximization through Adversarial Training](https://openreview.net/forum?id=HkeryxBtPB) (ICLR 2020) by [Gavin Weiguang Ding](http://gwding.github.io/), [Yash Sharma](https://www.yash-sharma.com/), [Kry Yik Chau Lui](https://www.linkedin.com/in/yik-chau-kry-lui-3887a955/?originalSubdomain=ca), and [Ruitong Huang](https://webdocs.cs.ualberta.ca/~ruitong/).

### Dependencies

- `pytorch (1.0.0)`
- `torchvision (0.2.1)`
- `advertorch (0.2.2)`

The code is tested with library versions specified above. It might also work with later versions.

### Overview of files

`anpgd.py` implements the AN-PGD attack used for MMA training.
`config.py` contains all the default training hyperparameters.
`utils.py` provides utility functions for MMA training.
`trainer.py` implements the MMA training algorithm.
`train.py` runs the MMA training process.
`evaluate_on_pgd_attacks.py`
`run_mnist_training.sh` and `run_cifar10_training.sh` contain commands for reproducing MMA models in the paper. 
`trained_models` contains pretrained MMA models.
`attack_mnist_models.sh` and `attack_cifar10_models.sh` contain command for evaluating MMA models with repeated whitebox PGD attacks.

### Examples

To train a MMA model on CIFAR10 with d_max=32/255 under Linf attacks, run
```
python train.py --dataset cifar10 --norm Linf --hinge_maxeps 0.1255 --seed 0 --savepath ./trained_models/cifar10-Linf-MMA-32-sd0
```

After training, to evaluate this model under Linf attacks with epsilon=8/255, run
```
python evaluate_on_pgd_attacks.py --dataset cifar10 --norm Linf --eps 0.0314 --seed 0 --model ./trained_models/cifar10-Linf-MMA-32-sd0/model_best.pt
```

See `run_mnist_training.sh` and `run_cifar10_training.sh` for the complete list.

### Reference

bibtex entry:
```bibtex
@inproceedings{
Ding2020MMA,
title={{MMA} Training: Direct Input Space Margin Maximization through Adversarial Training},
author={Ding, Gavin Weiguang and Sharma, Yash and Lui, Kry Yik Chau and Huang, Ruitong},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=HkeryxBtPB}
}
```

