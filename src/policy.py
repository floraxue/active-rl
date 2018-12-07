import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PolicyNet(nn.Module):
    def __init__(self, in_size, num_actions=2, feat_size=64,
                 hidden_size=128, nlayers=1):
        super(PolicyNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.num_layers = nlayers

        # feature dimension reduction
        self.conv = nn.Conv2d(in_size, feat_size, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(feat_size)
        self.relu = nn.ReLU(inplace=True)
        self.rnn = nn.LSTM(feat_size, hidden_size, num_layers=nlayers,
                           batch_first=True)
        self.proj = nn.Conv2d(hidden_size, feat_size * hidden_size,
                              kernel_size=1, bias=False)
        self.fc = nn.Conv2d(hidden_size, num_actions,
                            kernel_size=1, bias=False)

        self.hidden = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def init_hidden(self, bsz):
        h0 = torch.zeros(self.num_layers, bsz, self.hidden_size)
        c0 = torch.zeros(self.num_layers, bsz, self.hidden_size)
        return h0.cuda(), c0.cuda()

    def reset_hidden(self):
        self.hidden = None

    def forward(self, x):
        if len(x.size()) == 1:
            bsz = 1
            fsz = x.size(0)
        else:
            bsz, fsz = x.size(0), x.size(1)
        x = x.resize(bsz, fsz, 1, 1)
        # (1, 2048, 1, 1)
        orig_feat = x = self.conv(x)
        #(1, 64, 1, 1)
        # x = self.bn(x)
        x = self.relu(x)
        # sequence length is 1, (batch size, seq len, feat size)
        x = x.resize(bsz, 1, self.feat_size)
        #(1, 1, 64)
        if self.hidden is None:
            self.hidden = self.init_hidden(bsz)

        out, self.hidden = self.rnn(x, self.hidden)

        out = out.resize(bsz, self.hidden_size, 1, 1)
        out = self.proj(out)

        out = out.resize(bsz, self.hidden_size, self.feat_size)

        orig_feat = orig_feat.resize(bsz, self.feat_size, 1)

        out = torch.matmul(out, orig_feat)

        out = out.resize(bsz, self.hidden_size, 1, 1)
        out = self.fc(out).squeeze()

        return out, self.hidden