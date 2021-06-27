import torch
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import datasets
import torch.backends.cudnn as cudnn
import numpy as np
import os
from torch.utils.data import RandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

import pickle
import pdb

import time

import sys
sys.path.append('../pyvacy')
from pyvacy import analysis, sampling
from pyvacy import dpoptim
from torch.utils.data import TensorDataset

from lanczos import lanczos_tridiag_with_rop,eval_Mt_vec_prod_with_rop, lanczos_tridiag




def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--optim', default='DPSGD', type=str, help='optimizer',
                        choices=[ 'sgd', 'DPSGD'])
    parser.add_argument('--num-sample', type=int, default=10000,
                            help='training sample size')
    parser.add_argument('--num-valid', type=int, default=100,
                            help='validation sample size')
    parser.add_argument('--repeat', type=int, default=3,
                            help='number of repeated trainings')

    parser.add_argument('--micro-size', type=int, default=1,
                            help='micro-batch')
    parser.add_argument('--batch-size', type=int, default=250, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--l2_norm_clip', default=1.0, type=float, help='L2 sensitivity for dp')
    parser.add_argument('--stdev', default=4, type=float, help='std for dp')
    parser.add_argument('--delta', default=0.0001, type=float, help='delta for dp')

    parser.add_argument('--proj-epoch', type=int, default=100, metavar='N',
                        help='number of epochs to start projection (default: 10)')
    parser.add_argument('--proj-dim', type=int, default=20,
                            help='number of project dimensions')


    parser.add_argument('--arch', default='covnet', type=str, help='model',
                        choices=['covnet'])
    parser.add_argument('--data', default='mnist', type=str, help='data',
                        choices=[ 'mnist'])
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='weight decay for optimizers')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')


    return parser

def parameters_to_vector(parameters):
    r"""Convert parameters to one vector
    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    Returns:
        The parameters represented by a single vector
    """
    param_device = None

    vec = []
    for param in parameters:

        vec.append(param.grad.view(-1))
    return torch.cat(vec)

def param_to_vector(parameters):
    r"""Convert parameters to one vector
    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    Returns:
        The parameters represented by a single vector
    """
    param_device = None

    vec = []
    for param in parameters:

        vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_parameters(vec, parameters):
    r"""Convert one vector to the parameters
    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """

    param_device = None

    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        #param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.grad = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param



def add_noise(parameters,args):
    for p in parameters:
        p.grad.data.add_(args.l2_norm_clip * args.stdev * torch.randn_like(p.grad.data))
        p.grad.data.mul_(args.micro_size / args.batch_size)


def build_dataset(args):
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    print('==> Preparing data..')
    train_set = datasets.MNIST(
        '../data', train=True, download=True,
        transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        )

    test_set = datasets.MNIST(
        '../data', train=False, download=True,
        transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        )

    #split valid set from train set #
    num_train = len(train_set)
    indices = list(range(num_train))
    split = args.num_valid

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_set_new = datasets.SubData(train_set.data, train_set.targets)
    valid_set = datasets.SubData(train_set.data, train_set.targets)

    train_set_new.data = [train_set.data[idx] for idx in train_idx]
    train_set_new.targets = [train_set.targets[idx] for idx in train_idx]

    valid_set.data = [train_set.data[idx] for idx in valid_idx]
    valid_set.targets = [train_set.targets[idx] for idx in valid_idx]

    return train_set_new, valid_set, test_set


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        #return x
        return F.log_softmax(x,dim=1)


def create_optimizer(args, model_params):

    if args.optim == 'sgd':
        return optim.SGD(model_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)

    elif args.optim == 'DPSGD':
        return dpoptim.DPSGD(l2_norm_clip=args.l2_norm_clip, noise_multiplier=args.stdev, minibatch_size=args.batch_size, microbatch_size =args.micro_size,
                               params= model_params, lr = args.lr,  weight_decay=args.weight_decay)



def train_per_sample_clip(args, model, device, train_loader, valid_loader, optimizer, epoch):
    model.train()
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        mini_batch_loader = torch.utils.data.DataLoader(TensorDataset(data, target), batch_size=args.micro_size)
        for batch_idx, (data, target) in enumerate(mini_batch_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_microbatch_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.microbatch_step()
        # get clipped gradient # no noise in this step
        optimizer.noisy_grad()
        # add noise to the gradien#
        add_noise(model.parameters(),args)

        if epoch >= args.proj_epoch:
            #get the subspace of M_t on the valid dataset#
            Mteig,Mtq = lanczos_tridiag(args,model,valid_loader,device, args.proj_dim,sig2 = 0,isHessian = False)
            lamda, v = torch.symeig(Mteig, eigenvectors=True) #np.linalg.eigh(Mteig) # lambda in ascending order, so does v
            k = args.proj_dim
            v = torch.flip(v,[1])
            Vk = torch.mm(Mtq,v)
            Vk = Vk[:,:k]
            g = parameters_to_vector(model.parameters()).unsqueeze(-1).to(device)
            g_hat = torch.mm(Vk.transpose(0,1),g)
            g_hat = torch.mm(Vk,g_hat)
            #finish project
            vector_to_parameters(g_hat, model.parameters())

        optimizer.step()
        count = count +1

    train_loss = 0
    correct = 0
    with torch.no_grad():
        for  data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)
    print('epoch ', epoch)
    print('\nTrain loss: {}, Accuracy: {}'.format(train_loss, train_acc))
    return train_loss, train_acc



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

    train_loss = 0
    correct = 0
    with torch.no_grad():
        for  data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)
    print('\nTrain loss: {}, Accuracy: {}'.format(train_loss, train_acc))
    return train_loss, train_acc





def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest loss: {}, Accuracy: {}'.format(test_loss, test_acc))
    return test_loss, test_acc



def main():
    parser = get_parser()
    args = parser.parse_args()

    np.random.seed(0)

    use_cuda =  torch.cuda.is_available()

    device = 'cuda' if use_cuda else 'cpu'

    train_set, valid_set, test_set = build_dataset(args)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.test_batch_size)

    valid_loader = torch.utils.data.DataLoader(valid_set,
                 batch_size=1)

    train_set_sample = datasets.SubData(
        train_set.data, train_set.targets)

    stdev = args.stdev

    def one_run(args, train_sample_num):

        train_set_sample.set_sub_sample(train_sample_num)
        # compute the privacy cost using ma #
        print('Achieves ({}, {})-DP'.format(analysis.epsilon(len(train_set_sample),
            args.batch_size,
            args.stdev,
            args.epochs*len(train_set_sample)/args.batch_size,
            args.delta
        ),
        args.delta,
    ))
        train_loader = torch.utils.data.DataLoader(
            train_set_sample, batch_size=args.batch_size)

        model = SampleConvNet().to(device)

        optimizer  = create_optimizer(args, model.parameters())

        Tr_loss = []
        Tr_acc = []
        Te_loss = []
        Te_acc = []

        for epoch in range(1, args.epochs + 1):

            if "DP" in args.optim:
                train_loss, train_acc = train_per_sample_clip(args, model, device, train_loader, valid_loader,optimizer, epoch)

            else:
                train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch)

            test_loss, test_acc = test(args, model, device, test_loader)


            Tr_loss.append(train_loss)
            Tr_acc.append(train_acc)
            Te_loss.append(test_loss)
            Te_acc.append(test_acc)


        best_train_acc = max(Tr_acc)
        best_test_acc = max(Te_acc)
        best_train_loss = min(Tr_loss)
        best_test_loss = min(Te_loss)
        return best_train_acc, best_test_acc, best_train_loss, best_test_loss, Tr_acc, Te_acc, Tr_loss, Te_loss


    train_sample_nums = [args.num_sample]


    for train_sample_num in train_sample_nums:

        train_accs_hist = []
        test_accs_hist = []
        train_loss_hist = []
        test_loss_hist= []


        for no_repeat in range(args.repeat):
            torch.manual_seed(no_repeat)

            print('train sample num {}, repeat num {}'.format(
                train_sample_num, no_repeat))

            best_train_acc, best_test_acc, best_train_loss, best_test_loss, train_acc_all, test_acc_all, train_loss_all, test_loss_all\
                = one_run(args, train_sample_num)

            train_accs_hist.append(train_acc_all)
            test_accs_hist.append(test_acc_all)
            train_loss_hist.append(train_loss_all)
            test_loss_hist.append(test_loss_all)


    result = {'train_accs_hist': train_accs_hist, 'test_accs_hist': test_accs_hist,
              'train_loss_hist': train_loss_hist, 'test_loss_hist': test_loss_hist}


    file_path = "DPSGD_e{}_lr{}_clip{}_mc{}_dim{}_v{}".format(args.proj_epoch, args.lr, args.l2_norm_clip, args.micro_size, args.proj_dim, args.num_valid)

    file_name = 'result_proj_DPSGD_no{}_b{}_n{}'.format(args.stdev, args.batch_size, args.num_sample)

    file_dir = os.path.join("output", file_name)

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    file_path = os.path.join(file_dir, file_path)

    with open(file_path, 'wb') as fou:
        pickle.dump(result, fou)


if __name__ == '__main__':
    main()
