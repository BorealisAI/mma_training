python train.py --dataset cifar10 --norm Linf --hinge_maxeps 0.0471 --seed 0 --savepath ./trained_models/cifar10-Linf-MMA-12-sd0

python train.py --dataset cifar10 --norm Linf --hinge_maxeps 0.0784 --seed 0 --savepath ./trained_models/cifar10-Linf-MMA-20-sd0

python train.py --dataset cifar10 --norm Linf --hinge_maxeps 0.1255 --seed 0 --savepath ./trained_models/cifar10-Linf-MMA-32-sd0

python train.py --dataset cifar10 --norm Linf --hinge_maxeps 0.0471 --seed 0  --clean_loss_coeff 0. --savepath ./trained_models/cifar10-Linf-OMMA-12-sd0

python train.py --dataset cifar10 --norm Linf --hinge_maxeps 0.0784 --seed 0  --clean_loss_coeff 0. --savepath ./trained_models/cifar10-Linf-OMMA-20-sd0

python train.py --dataset cifar10 --norm Linf --hinge_maxeps 0.1255 --seed 0  --clean_loss_coeff 0. --savepath ./trained_models/cifar10-Linf-OMMA-32-sd0


python train.py --dataset cifar10 --norm L2 --hinge_maxeps 1. --seed 0 --savepath ./trained_models/cifar10-L2-MMA-1.0-sd0

python train.py --dataset cifar10 --norm L2 --hinge_maxeps 2. --seed 0 --savepath ./trained_models/cifar10-L2-MMA-2.0-sd0

python train.py --dataset cifar10 --norm L2 --hinge_maxeps 3. --seed 0 --savepath ./trained_models/cifar10-L2-MMA-3.0-sd0

python train.py --dataset cifar10 --norm L2 --hinge_maxeps 1. --seed 0  --clean_loss_coeff 0. --savepath ./trained_models/cifar10-L2-OMMA-1.0-sd0

python train.py --dataset cifar10 --norm L2 --hinge_maxeps 2. --seed 0  --clean_loss_coeff 0. --savepath ./trained_models/cifar10-L2-OMMA-2.0-sd0

python train.py --dataset cifar10 --norm L2 --hinge_maxeps 3. --seed 0  --clean_loss_coeff 0. --savepath ./trained_models/cifar10-L2-OMMA-3.0-sd0
