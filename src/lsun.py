import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import time
import argparse
from dataset import ImageData
import network
from util import logger, softmax
from env_obj import Env
import os
from os.path import join, exists
import shutil
import pdb
import torch.nn.functional as F
import numpy as np
import json
import pickle
from collections import namedtuple

Sample = namedtuple('Sample', ('feat', 'label', 'key'))

def train_lsun_model(game):
    #load keys

    #load model


def test_lsun_model():
    #load keys

    # load model

def load_keys(game, iter, curr_k):
    category = 'cat'
    train_prefix = '{}_episode_{:04d}_update_{:03d}_LSUN'.format(
        category, iter, curr_k)
    work_root = '/data/active-rl-data/classifier'
    work_dir = join(work_root, train_prefix)

    train_keys_path = join(work_dir, 'loader', 'train_keys.p')
    val_keys_path = join(work_dir, 'loader', 'val_keys.p')
    past_train_keys_path = join(work_root, 'past_train_keys.p')
    past_val_keys_path = join(work_root, 'past_val_keys.p')

    # Load past train val keys
    if exists(past_train_keys_path):
        past_train_keys = pickle.load(open(past_train_keys_path, 'rb'))
    else:
        past_train_keys = []
    if exists(past_val_keys_path):
        past_val_keys = pickle.load(open(past_val_keys_path, 'rb'))
    else:
        past_val_keys = []

    # Balance labels for current train val set
    train_data= map(lambda x: Sample(*x), game.train_data[game.order[game.last: game.index]])
    chosen_data = np.random.choice(train_data, game.latest_num_train)

    train_keys = game.chosen_train['key']
    train_labels = game.chosen_train['gt']
    val_keys = game.chosen_val['key']
    val_labels = game.chosen_val['gt']

    train_order = game.balance_labels(train_labels)
    train_keys_curr = [train_keys[i] for i in train_order]
    val_bal_order = game.balance_labels(val_labels)
    val_bal_keys_curr = [val_keys[i] for i in val_bal_order]

    # TODO: assume we don't need val keys before bal
    train_keys_all = train_keys_curr + past_train_keys
    val_keys_all = val_bal_keys_curr + past_val_keys

    pickle.dump(train_keys_all, open(train_keys_path, 'wb'))
    pickle.dump(val_keys_all, open(val_keys_path, 'wb'))

    # Save past train val keys for next time to use
    pickle.dump(train_keys_all, open(past_train_keys_path, 'wb'))
    pickle.dump(val_keys_all, open(past_val_keys_path, 'wb'))

    save_dir = join(work_dir, 'snapshots')
    os.makedirs(save_dir, exist_ok=True)
    method = 'resnet'

    # TODO set iters?
    if self.terminal:
        iters = 15000
    else:
        iters = 5000