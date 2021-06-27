This repository is the implementation of "Bypassing the Ambient Dimension: Private SGD with Gradient Subspace Identification".

The implementation is in python.

The folder 'fashion-mnist-cnn' includes the code for Fashion MNIST experiment.
run_nn_fmnist_sgd_proj.py is the main file.

You can run the DPD-SGD by the command:

python3 run_nn_fmnist_sgd_proj.py  --stdev 18 --batch-size  250 --lr 0.01 --micro-size 5 --proj-dim 70 --num-valid 100 --proj-epoch 15

The folder 'mnist-cnn' includes the code for the MNIST experiment.
run_nn_mnist_sgd_proj.py is the main file.

python3 run_nn_mnist_sgd_proj.py  --stdev 14 --batch-size  250 --lr 0.2 --micro-size 1 --proj-dim 50 --num-valid 100 --proj-epoch 0

Parameters:

'--stdev': noise variance

'--batch-size': batch size

'--lr': learning rate

'proj-dim': projection dimension, i.e., k in the DPD-SGD

'--num-valid': public dataset size

' --proj-epoch': number of epochs to start projection
