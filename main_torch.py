"""
hello!

"""

import sys

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
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
                       grad_noise=True):
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
        for i in range(n_layer):
            self.model.add_module("fc".format(i), nn.Linear(hidden_dim, hidden_dim))
            self.model.add_module("relu".format(i), nn.ReLU())

        self.model.add_module("output", nn.Linear(hidden_dim, output_dim))

        self.opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

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

            if self.grad_noise:
                loss.register_hook(lambda g : g + noise)

            loss.backward()
            self.opt.step()

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


    def add_noise(self, arg1):
        """TODO: Docstring for add_noise.

        Parameters
        ----------
        arg1 : TODO

        Returns
        -------
        TODO

        """
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
    n_epoch = 20
    batch_size = 2**6
    set_gpu = True


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
                  grad_noise=True)

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
    - [ ] init, simple, bad, good
    - [ ] add dropout
    - [ ] grad clipping
    - [ ] cross validate with hyperparm

    """
    main()
