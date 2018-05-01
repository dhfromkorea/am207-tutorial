"""
hello!

"""

import os
import sys
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dsets
from torch.nn.utils import clip_grad_norm
import torchvision.transforms as transforms

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("bmh")


DATA_PATH = "data"

class DeepFFN(nn.Module):
    """20 hidden-layer deep feedforward network """

    def __init__(self, input_dim,
                       hidden_dim,
                       output_dim,
                       n_layer,
                       learning_rate,
                       set_gpu,
                       grad_noise,
                       gamma,
                       eta,
                       grad_clip,
                       grad_clip_norm,
                       grad_clip_value,
                       init_weight_type,
                       simple_init_std,
                       weight_decay,
                       debug):
        """TODO: to be defined1.

        Parameters
        ----------
        arg1 : TODO
        arg2 : TODO


        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.set_gpu = set_gpu
        self.debug = debug
        self.simple_init_std = simple_init_std

        self.model = nn.Sequential()
        self.model.add_module("input", nn.Linear(input_dim, hidden_dim))
        for i in range(n_layer - 1):
            self.model.add_module("fc{}".format(i+1), nn.Linear(hidden_dim, hidden_dim))
            self.model.add_module("relu{}".format(i+1), nn.ReLU())
        self.model.add_module("fc{}".format(n_layer), nn.Linear(hidden_dim, output_dim))

        self._init_weight_type = init_weight_type
        self.model.apply(self.initialize_weight)


        self.opt = torch.optim.SGD(self.model.parameters(),
                                   lr=learning_rate,
                                   weight_decay=weight_decay)
        # output layer implicitly defined by this
        self.criterion = nn.CrossEntropyLoss()

        self._step = 0
        self.grad_noise = grad_noise
        self._gamma = gamma
        self._eta = eta

        if self.grad_noise:
            for m in self.model.modules():
                classname = m.__class__.__name__
                if classname.find("Linear") != -1:
                    m.register_backward_hook(self.add_grad_noise)



        self._grad_clip = grad_clip
        self._grad_clip_norm = grad_clip_norm
        self._grad_clip_value = grad_clip_value

        if set_gpu:
            self.cuda()



    def train(self, epoch, train_loader):
        """TODO: Docstring for train.

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """
        self.model.train()
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            if self.set_gpu:
                data, target = Variable(data).cuda(), Variable(target).cuda()
            else:
                data, target = Variable(data), Variable(target)
            data = data.view(-1, self.input_dim)

            self.opt.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()

            if self._grad_clip:
                clip_grad_norm(self.model.parameters(), self._grad_clip_value, self._grad_clip_norm)


            if self.debug and batch_idx % 10 == 0:
                for layer in self.model.modules():
                    if np.random.rand() > 0.95:
                        if isinstance(layer, nn.Linear):
                            weight = layer.weight.data.numpy()
                            print("========================")
                            print("weight\n")
                            print("max:{}\tmin:{}\tavg:{}\n".format(weight.max(), weight.min(), weight.mean()))
                            grad = layer.weight.grad.data.numpy()
                            print("grad\n")
                            print("max:{}\tmin:{}\tavg:{}\n".format(grad.max(), grad.min(), grad.mean()))
                            print("=========================")

            self.opt.step()
            self._step += 1

            if np.isnan(loss.data[0]):
                raise Exception("gradient exploded or vanished: try clipping gradient")


            if batch_idx % 100 == 0:
                sys.stdout.flush()
                sys.stdout.write('\rTrain Epoch: {:<2} [{:<5}/{:<5} ({:<2.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))

            total_loss += loss.data[0]

        return total_loss/len(train_loader)


    def validate(self, val_loader):
        """TODO: Docstring for function.

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """
        # set self.training to False
        # e.g. this will stip off softmax
        self.model.eval()
        val_loss = 0
        correct = 0

        for data, target in val_loader:
            if self.set_gpu:
                data, target = Variable(data).cuda(), Variable(target).cuda()
            else:
                data, target = Variable(data), Variable(target)
            data = data.view(-1, self.input_dim)
            output = self.model(data)
            loss_ = self.criterion(output, target).data[0]
            val_loss += loss_ # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum()

        val_loss /= len(val_loader)

        accuracy = 100. * correct / len(val_loader.dataset)

        sys.stdout.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n\n'.format(
            val_loss, correct, len(val_loader.dataset),
            accuracy))
        return val_loss, accuracy


    def _compute_grad_noise(self, grad):
        """TODO: Docstring for function.

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """

        std = np.sqrt(self._eta / (1 + self._step)**self._gamma)
        return Variable(grad.data.new(grad.size()).normal_(0, std=std))


    def add_grad_noise(self, module, grad_i_t, grad_o):
        """TODO: Docstring for add_noise.

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """
        _, _, grad_i = grad_i_t[0], grad_i_t[1], grad_i_t[2]
        noise = self._compute_grad_noise(grad_i)
        return (grad_i_t[0], grad_i_t[1], grad_i + noise)


    def initialize_weight(self, module):
        """TODO: Docstring for initialize_weight.

        Parameters
        ----------
        module : TODO

        Returns
        -------
        TODO

        """
        if type(module) == nn.Linear:
            if self._init_weight_type == "good":
                # he 2015
                torch.nn.init.kaiming_normal(module.weight, mode='fan_out')
                # it seems you don't initialize bias
            elif self._init_weight_type == "bad":
                # zero init
                module.weight.data.fill_(0)
                #module.bias.data.zero_()
            elif self._init_weight_type == "simple":
                # gaussian (0, 0.1^2)
                torch.nn.init.normal(module.weight, mean=0, std=self.simple_init_std)
                #torch.nn.init.normal(module.bias, mean=0, std=0.1)
            else:
                pass


def main(args):
    """TODO: Docstring for main.

    Parameters
    ----------
    arg1 : TODO


    Returns
    -------
    TODO

    """

    np.random.seed(0)
    torch.manual_seed(0)

    # hyperparam
    n_epoch = args.n_epoch
    batch_size = args.batch_size
    set_gpu = args.cuda
    eta = args.eta
    gamma = args.gamma

    grad_clip = args.grad_clip
    grad_clip_norm = args.grad_clip_norm
    grad_clip_value = args.grad_clip_value

    init_weight_type = args.init_weight_type
    grad_noise = args.grad_noise

    exp_id = args.exp_id


    # prepare data
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)


    val_dataset = dsets.MNIST(root='./data',
                               train=False,
                               transform=transforms.ToTensor(),
                               download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=args.n_worker)


    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=len(val_dataset),
                                             shuffle=False,
                                             num_workers=args.n_worker)

    ffn = DeepFFN(input_dim=28*28,
                  hidden_dim=50,
                  output_dim=10,
                  n_layer=20,
                  learning_rate=args.lr,
                  set_gpu=set_gpu,
                  grad_noise=grad_noise,
                  gamma=gamma,
                  eta=eta,
                  grad_clip=grad_clip,
                  grad_clip_norm=grad_clip_norm,
                  grad_clip_value=grad_clip_value,
                  init_weight_type=init_weight_type,
                  simple_init_std=args.simple_init_std,
                  weight_decay=args.weight_decay,
                  debug=args.debug)

    res = {}
    res["exp_id"] = args.exp_id
    res["n_epoch"] = args.n_epoch
    res["train_loss"] = []
    res["val_loss"] = []
    res["val_accuracy"] = []
    for epoch in range(n_epoch):
        #loss_t, loss_list_t = ffn.train(epoch, train_loader)
        loss_t = ffn.train(epoch, train_loader)
        res["train_loss"].append(loss_t)

        loss_v, acc_v = ffn.validate(val_loader)
        res["val_loss"].append(loss_v)
        res["val_accuracy"].append(acc_v)

    plot_accuracy(res, show=False)

    save_path = os.path.join(DATA_PATH, exp_id + ".pkl")
    with open(save_path, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

def plot_accuracy(res, show=True):
    # train loss
    n_epoch, exp_id = res["n_epoch"], res["exp_id"]
    plt.figure(figsize=(7,7))
    plt.title("Training: {}".format(exp_id), fontsize=15.0)
    plt.plot(range(1, n_epoch + 1), res["train_loss"], c="r", linestyle="--", marker="o")
    plt.xlabel("Epoch", fontsize=15.0)
    plt.ylabel("Loss", fontsize=15.0)


    if show:
        plt.show()

    plt.savefig("data/{}_train_loss.png".format(exp_id), format="png", bbox_inches="tight")
    plt.close()

    # validation accuracy

    plt.figure(figsize=(7,7))
    plt.title("Validation: {}".format(exp_id), fontsize=15.0)
    plt.plot(range(1, n_epoch + 1), res["val_accuracy"], c="b", linestyle="--", marker="o")
    plt.xlabel("Epoch", fontsize=15.0)
    plt.ylabel("Accuracy", fontsize=15.0)

    if show:
        plt.show()

    plt.savefig("data/{}_val_accuracy.png".format(exp_id), format="png", bbox_inches="tight")
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--n_epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.01')
    parser.add_argument('--grad_noise', action='store_true', help='add gradient noise')
    parser.add_argument('--eta', type=float, default=0.01, help='eta')
    parser.add_argument('--gamma', type=float, default=0.55, help='set gamme for guassian noise')
    parser.add_argument('--grad_clip', action='store_true', help='clip gradient')
    parser.add_argument('--grad_clip_norm', type=int, default=2, help='norm of the gradient clipping, default: l2')
    parser.add_argument('--grad_clip_value', type=float, default=10.0, help='the gradient clipping value')
    parser.add_argument('--init_weight_type', type=str, default="good", choices=["good", "simple", "bad"], help='weight init scheme')
    parser.add_argument('--exp_id', type=str, default="", help='experiment id')
    parser.add_argument('--n_worker', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--debug', action='store_true', help='enables debug mode')
    parser.add_argument('--simple_init_std', type=float, default=0.1, help='std for simple init')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for sgd')
    parser.add_argument('--outf', default='data', help='folder to output images and model checkpoints')
    return parser


if __name__ == "__main__":
    """
    todo
    - [v] cpu
    - [v] gpu compatible
    - [v] add gradient noise
    - [v] init, simple, bad, good
    - [ ] add dropout
    - [v] grad clipping (both by value and norm)
    - [ ] cross validate with hyperparm
    - [v] add argparser for experiments
    - [ ] monitor grad
    - [ ] check simple and bad init if correctly implemented


    problems

    - too slow
    - no gradient clipping -> vanishing and exploding...

    """

    args = parse_args().parse_args()



    print("hello world! \n")

    for arg in vars(args):
        print(arg, getattr(args, arg))

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    print("\ngood luck! \n")
    main(args)
