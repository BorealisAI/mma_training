python evaluate_on_pgd_attacks.py --dataset mnist --norm Linf --eps 0.3 --model ./trained_models/mnist-Linf-MMA-0.45-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset mnist --norm Linf --eps 0.3 --model ./trained_models/mnist-Linf-OMMA-0.45-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset mnist --norm L2 --eps 2. --model ./trained_models/mnist-L2-MMA-2.0-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset mnist --norm L2 --eps 2. --model ./trained_models/mnist-L2-MMA-4.0-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset mnist --norm L2 --eps 2. --model ./trained_models/mnist-L2-MMA-6.0-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset mnist --norm L2 --eps 2. --model ./trained_models/mnist-L2-OMMA-2.0-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset mnist --norm L2 --eps 2. --model ./trained_models/mnist-L2-OMMA-4.0-sd0/model_best.pt

python evaluate_on_pgd_attacks.py --dataset mnist --norm L2 --eps 2. --model ./trained_models/mnist-L2-OMMA-6.0-sd0/model_best.pt
