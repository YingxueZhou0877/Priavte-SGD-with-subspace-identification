import sys
sys.path.append('../pyvacy')
from pyvacy import analysis, sampling

import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--optim', default='sagd', type=str, help='optimizer',
                        choices=[ 'sgd', 'adam','SARMSprop','SAGD', 'SAdam','DPSGD','DPRMSprop','DPRMSprop'])
    parser.add_argument('--repeat', type=int, default=1,
                            help='number of repeated trainings')
    parser.add_argument('--batch-size', type=int, default=250, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--noise-coe', type=float, default=1, metavar='NO',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--l2_norm_clip', default=1, type=float, help='L2 sensitivity for dp')
    parser.add_argument('--stdev', default=8, type=float, help='std for dp')
    parser.add_argument('--delta', default=0.00001, type=float, help='delta for dp')




    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--LR-decay',  default= 'False',
                        help='Decay learning rate by epoch', choices=['False', 'True'])
    parser.add_argument('--decay-epoch', type=int, default=5, metavar='N',
                        help='number of epochs to decay (default: 10)')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')


    return parser

def main():

    parser = get_parser()
    args = parser.parse_args()

    train_sample_num = 50000


    print('Achieves ({}, {})-DP'.format(analysis.epsilon(train_sample_num,
                args.batch_size,
                args.stdev,
                args.epochs*train_sample_num/args.batch_size,
                args.delta
            ),
            args.delta,
        ))



if __name__ == '__main__':
    main()
