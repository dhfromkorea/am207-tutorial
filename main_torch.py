"""
hello!

"""

import sys

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class DeepFFN(nn.Module):

    """20 hidden-layer deep feedforward network """

    def __init__(self, input_dim,
                       hidden_dim,
                       output_dim,
                       n_layer=20,
                       learning_rate=1e-3,
                       set_gpu=False,
                       grad_noise=True,
                       gamma=0.55,
                       eta=0.3,
                       grad_clip=False,
                       grad_clip_norm=2,
                       grad_clip_value=100.0,
                       init_weight_type="good"
                       ):
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

        self.model = nn.Sequential()
        self.model.add_module("input", nn.Linear(input_dim, hidden_dim))
        for i in range(n_layer - 1):
            self.model.add_module("fc{}".format(i+1), nn.Linear(hidden_dim, hidden_dim))
            self.model.add_module("relu{}".format(i+1), nn.ReLU())
        self.model.add_module("fc{}".format(n_layer), nn.Linear(hidden_dim, output_dim))

        self._init_weight_type = init_weight_type
        self.model.apply(self.initialize_weight)



        self.opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
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

            self.opt.step()
            self._step += 1


            if batch_idx % 100 == 0:
                sys.stdout.flush()
                sys.stdout.write('\rTrain Epoch: {:<2} [{:<5}/{:<5} ({:<2.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
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
        misclassified_samples = []

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
            pred_label = pred.cpu().numpy()[0][0]
            target_label = target.cpu().data.numpy()[0]

            if pred_label != target_label:
                misclassified = (data, pred_label, target_label)
                misclassified_samples.append(misclassified)

            correct += pred.eq(target.data.view_as(pred)).sum()
        val_loss /= len(val_loader.dataset)

        sys.stdout.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n\n'.format(
            val_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
        return val_loss, misclassified_samples


    def _compute_grad_noise(self, grad):
        """TODO: Docstring for function.

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """

        # make this a variable
        std = self._eta / (1 + self._step)**self._gamma
        print("std", std)
        print("step", self._step)
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
        print("noise avg", noise.mean())
        print("grad avg", grad_i.mean())
        print("grad norm", torch.norm(grad_i, self._grad_clip_norm))
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
        classname = module.__class__.__name__
        if classname.find("Linear") != -1:
            if self._init_weight_type == "good":
                # he 2015
                torch.nn.init.kaiming_normal(module.weight, mode='fan_out')
                # it seems you don't initialize bias
            elif self._init_weight_type == "bad":
                # zero init
                module.weight.data.fill_(0)
                module.bias.data.zero_()
            elif self._init_weight_type == "simple":
                # gaussian (0, 0.1^2)
                module.weight.data.normal_(0, 0.1)
                module.bias.data.normal_(0, 0.1)
            else:
                pass


def main():
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
    n_epoch = 500
    batch_size = 60000
    set_gpu = True
    eta_list = [0.01, 0.3, 1.0]
    eta = eta_list[0]
    gamma = 0.55

    grad_clip = True
    grad_clip_norm = 2
    #grad_clip_norm = float("inf")
    #grad_clip_norm = 1
    grad_clip_value = 10.0

    init_weight_type = "good"
    grad_noise = True

    if batch_size == 60000:
        print("GD")
    else:
        print("SGD")

    if grad_noise:
        print("using noise")
    else:
        print("NOT using noise")





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
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=1,
                                              shuffle=False)

    ffn = DeepFFN(input_dim=28*28,
                  hidden_dim=50,
                  output_dim=10,
                  n_layer=20,
                  learning_rate=1e-2,
                  set_gpu=set_gpu,
                  grad_noise=grad_noise,
                  gamma=gamma,
                  eta=eta,
                  grad_clip=grad_clip,
                  grad_clip_norm=grad_clip_norm,
                  grad_clip_value=grad_clip_value,
                  init_weight_type=init_weight_type)

    res = {}
    res["loss_train"] = []
    res["loss_val"] = []
    for epoch in range(1, n_epoch + 1):
        loss_t = ffn.train(epoch, train_loader)
        res["loss_train"].append(loss_t)

        loss_v, misclassified_samples = ffn.validate(val_loader)
        res["loss_val"].append(loss_v)



if __name__ == "__main__":
    """
    todo
    - [v] cpu
    - [v] gpu compatible
    - [v] add gradient noise
    - [v] init, simple, bad, good
    - [ ] add dropout
    - [v] grad clipping (both by value and norm)
    - [ ] monitor grad
    - [ ] cross validate with hyperparm

    """
    main()
