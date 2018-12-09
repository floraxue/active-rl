#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math

__dict__ = ['mlp', 'resnet']


class MLP(nn.Module):
    def __init__(self, in_dim=2049, out_dim=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.5, inplace=True)
        self.fc2 = nn.Linear(128, out_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, x):
        x = x.squeeze()
        x = self.drop(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def optim_parameters(self):
        for param in self.parameters():
            yield param

class BinaryModel(nn.Module):
    """ :returns
            binary logits, confidence score in softmax or entropy value """

    def __init__(self, arch, pretrained=True, uncertain='prob'):
        super(BinaryModel, self).__init__()
        model = models.__dict__[arch](pretrained=pretrained)
        self.base = nn.Sequential(*list(model.children())[:-1])
        # only for resnet
        self.classify = nn.Conv2d(model.fc.in_features, 2,
                                  kernel_size=1, bias=True)

        # init weights
        m = self.classify
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()

        # uncertainty type: either probability or entropy
        self.uncertain = uncertain

    def forward(self, x):
        x = self.base(x)
        x = self.classify(x).squeeze()
        return x, None
        # if self.uncertain == 'entropy':
        #     prob = F.softmax(x, dim=-1)
        #     log_prob = F.log_softmax(x, dim=-1)
        #     uncertain = -1.0 * (prob * log_prob).sum(dim=-1)
        # else:
        #     uncertain = F.softmax(x, dim=-1)[:, -1]
        #  use the prob of class 1
        # return x, uncertain

    def optim_parameters(self):
        for param in self.base.parameters():
            yield param
        for param in self.classify.parameters():
            yield param


def resnet():
    model = BinaryModel('resnet18')
    return model
