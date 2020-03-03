python train.py --dataset mnist --norm Linf --hinge_maxeps 0.45 --seed 0 --savepath ./trained_models/mnist-Linf-MMA-0.45-sd0

python train.py --dataset mnist --norm Linf --hinge_maxeps 0.45 --seed 0 --clean_loss_coeff 0. --savepath ./trained_models/mnist-Linf-OMMA-0.45-sd0


python train.py --dataset mnist --norm L2 --hinge_maxeps 2. --seed 0 --savepath ./trained_models/mnist-L2-MMA-2.0-sd0

python train.py --dataset mnist --norm L2 --hinge_maxeps 4. --seed 0 --savepath ./trained_models/mnist-L2-MMA-4.0-sd0

python train.py --dataset mnist --norm L2 --hinge_maxeps 6. --seed 0 --savepath ./trained_models/mnist-L2-MMA-6.0-sd0

python train.py --dataset mnist --norm L2 --hinge_maxeps 2. --seed 0  --clean_loss_coeff 0. --savepath ./trained_models/mnist-L2-OMMA-2.0-sd0

python train.py --dataset mnist --norm L2 --hinge_maxeps 4. --seed 0  --clean_loss_coeff 0. --savepath ./trained_models/mnist-L2-OMMA-4.0-sd0

python train.py --dataset mnist --norm L2 --hinge_maxeps 6. --seed 0  --clean_loss_coeff 0. --savepath ./trained_models/mnist-L2-OMMA-6.0-sd0
