python evaluate_on_pgd_attacks.py --dataset cifar10 --norm Linf --eps 0.0314 --model ./trained_models/cifar10-Linf-MMA-12-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset cifar10 --norm Linf --eps 0.0314 --model ./trained_models/cifar10-Linf-MMA-20-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset cifar10 --norm Linf --eps 0.0314 --model ./trained_models/cifar10-Linf-MMA-32-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset cifar10 --norm Linf --eps 0.0314  --model ./trained_models/cifar10-Linf-OMMA-12-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset cifar10 --norm Linf --eps 0.0314  --model ./trained_models/cifar10-Linf-OMMA-20-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset cifar10 --norm Linf --eps 0.0314  --model ./trained_models/cifar10-Linf-OMMA-32-sd0/model_best.pt


python evaluate_on_pgd_attacks.py --dataset cifar10 --norm L2 --eps 1. --model ./trained_models/cifar10-L2-MMA-1.0-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset cifar10 --norm L2 --eps 1. --model ./trained_models/cifar10-L2-MMA-2.0-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset cifar10 --norm L2 --eps 1. --model ./trained_models/cifar10-L2-MMA-3.0-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset cifar10 --norm L2 --eps 1.  --model ./trained_models/cifar10-L2-OMMA-1.0-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset cifar10 --norm L2 --eps 1.  --model ./trained_models/cifar10-L2-OMMA-2.0-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset cifar10 --norm L2 --eps 1.  --model ./trained_models/cifar10-L2-OMMA-3.0-sd0/model_best.pt
