import numpy as np
import subprocess
from collections import namedtuple
import os
from os.path import join, exists
from train_new import train, test
from torchvision import models, transforms
import sys
import torch

from util import logger, checkdir
from dataset import FeatDataset, ImageData
import pickle
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
from path_constants import *
from train_new import USE_PRETRAINED

Sample = namedtuple('Sample', ('feat', 'label', 'key'))


class VFGGAME:

    def __init__(self, args):
        self.category = args.category
        self.key_path = args.key_path
        self.feat_dir = args.feat_dir
        self.image_dir = args.image_dir
        self.gt_path = args.gt_path
        self.initial_train_keys_path = args.initial_train_keys_path
        self.classifier_root = args.classifier_root
        self.train_data = FeatDataset(key_path=self.key_path,
                                      feat_dir=self.feat_dir,
                                      gt_path=self.gt_path)

        self.image_data = ImageData(key_path=self.key_path,
                                    image_dir=self.image_dir,
                                    gt_path=self.gt_path)

        # decide the querying order
        self.order = np.random.permutation(range(len(self.train_data)))
        # header of the training set
        self.index = 0

        self.budget = args.budget

        self.duration = args.duration
        # count the number of chosen samples
        self.chosen = 0
        self.chosen_set = {'key': [], 'gt': []}
        self.update = 0

        # holder for the current feature
        self.current_sample = None
        self.current_reward = 0
        # count the number of environment reset
        self.episode = 0
        self.terminal = False

        # num for setting train val and test set
        self.val_rate = args.val_rate
        self.test_rate = args.test_rate
        self.chosen_train = {'key': [], 'gt': []}
        self.chosen_val = {'key': [], 'gt': []}
        self.chosen_test = {'key': [], 'gt': []}

        #added
        # self.last = 0

    def reset(self, new_key_path):
        self.chosen = 0
        self.chosen_set = {'key': [], 'gt': []}
        self.chosen_train = {'key': [], 'gt': []}
        self.chosen_val = {'key': [], 'gt': []}
        self.chosen_test = {'key': [], 'gt': []}
        self.update = 0
        self.terminal = False
        self.episode += 1
        # shuffle the training order
        self.key_path = new_key_path
        self.train_data = FeatDataset(key_path=self.key_path,
                                      feat_dir=self.feat_dir,
                                      gt_path=self.gt_path)
        self.image_data = ImageData(key_path=self.key_path,
                                    image_dir=self.image_dir,
                                    gt_path=self.gt_path)
        self.order = np.random.permutation(range(len(self.train_data)))
        self.index = 0
        self.current_sample = None
        self.current_reward = 0

    def sample(self):
        """ return the state of the current image """
        item = self.train_data[self.order[self.index]]
        self.current_sample = Sample(*item)
        feat = np.array(self.current_sample.feat)
        img = self.image_data[self.order[self.index]][0]
        mean = np.array([0.49911337, 0.46108112, 0.42117174])
        std = np.array([0.29213443, 0.2912565, 0.2954716])

        normalize = transforms.Normalize(mean=mean, std=std)
        trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            normalize])

        prob = self.get_prob(trans(img)).data.cpu().numpy()
        x = np.concatenate((feat, prob))
        return x

    def get_prob(self, image):
        curr_model_path = join(self.classifier_root, "snapshots",
                               'bal_model_best.pth.tar')
        model = models.resnet18(pretrained=False, num_classes=2)
        model = torch.nn.DataParallel(model).cuda()
        if exists(curr_model_path):
            checkpoint = torch.load(curr_model_path)["state_dict"]
            model.load_state_dict(checkpoint)
        elif USE_PRETRAINED:
            checkpoint = model_zoo.load_url(model_urls['resnet18'])
            checkpoint.pop('fc.weight')
            checkpoint.pop('fc.bias')
            for k in list(checkpoint.keys())[:]:
                checkpoint['module.'+k] = checkpoint[k]
                checkpoint.pop(k)
            a = model.state_dict()
            a.update(checkpoint)
            model.load_state_dict(a)

        model.eval()
        output = model(image.unsqueeze(0))
        softmax_ = F.softmax(output, dim=-1)
        if len(softmax_.size()) == 2:
            softmax_ = softmax_[0]
        return softmax_

    def step(self, action):
        if self.chosen == self.budget or self.index == len(self.train_data) - 2:
            self.terminal = True

        if action > 0:
            # take the current sample
            self.chosen += 1
            self.chosen_set['key'] += [self.current_sample.key]
            self.chosen_set['gt'] += [self.current_sample.label]

            if self.chosen % self.duration == 0 or self.chosen == self.budget:
                # We have reached K, prepare for agent and classifier update
                curr_iter_num = self.chosen - ((self.chosen - 1) // self.duration) * self.duration
                num_val = int(self.val_rate * curr_iter_num)
                num_test = int(self.test_rate * curr_iter_num)
                num_train = curr_iter_num - num_val - num_test
                self.chosen_train['key'] += self.chosen_set['key'][:num_train]
                self.chosen_train['gt'] += self.chosen_set['gt'][:num_train]
                self.chosen_val['key'] += self.chosen_set['key'][num_train: num_train + num_val]
                self.chosen_val['gt'] += self.chosen_set['gt'][num_train: num_train + num_val]
                self.chosen_test['key'] += self.chosen_set['key'][-num_test:]
                self.chosen_test['gt'] += self.chosen_set['gt'][-num_test:]

                # #recording this for LSUN
                # self.latest_num_train = num_train
                # self.latest_num_val = num_val
                # self.latest_num_test = num_test
                self.update += 1

        # move to the next sample in the queue
        self.index += 1
        next_state = self.sample()

        return self.current_reward, next_state, self.terminal

    def train_model(self, lr, is_holdout):
        category = self.category
        # train_prefix = 'RL_{}_episode_{:04d}_update_{:03d}'.format(
        #     category, self.episode, self.update)

        train_keys_path = join(checkdir(join(self.classifier_root, 'loader')),
                               'train_keys.p')
        val_keys_path = join(checkdir(join(self.classifier_root, 'loader')),
                             'val_keys.p')
        past_train_keys_path = join(self.classifier_root, 'past_train_keys.p')
        past_val_keys_path = join(self.classifier_root, 'past_val_keys.p')

        # Load past train val keys
        if exists(past_train_keys_path):
            past_train_keys = pickle.load(open(past_train_keys_path, 'rb'))
        else:
            past_train_keys = pickle.load(open(self.initial_train_keys_path, 'rb'))

        if exists(past_val_keys_path):
            past_val_keys = pickle.load(open(past_val_keys_path, 'rb'))
        else:
            past_val_keys = []

        # Balance labels for current train val set
        train_keys_curr = self.chosen_train['key']
        val_bal_keys_curr = self.chosen_val['key']
        train_keys_all = train_keys_curr + past_train_keys
        val_keys_all = val_bal_keys_curr + past_val_keys

        # Save past train val keys for next time to use
        pickle.dump(train_keys_all, open(past_train_keys_path, 'wb'))
        pickle.dump(val_keys_all, open(past_val_keys_path, 'wb'))

        # Use only a portion of the past_train_keys to avoid overfit
        past_train_keys_curr = np.random.choice(past_train_keys,
                                                size=len(past_train_keys)//5,
                                                replace=False).tolist()
        pickle.dump(train_keys_curr + past_train_keys_curr, open(train_keys_path, 'wb'))
        pickle.dump(val_keys_all, open(val_keys_path, 'wb'))

        save_dir = checkdir(join(self.classifier_root, 'snapshots'))
        model_file_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        method = 'resnet'
        image_dir = IMAGE_DIR_HOLDOUT if is_holdout else IMAGE_DIR_TRAIN
        gt_path = GT_PATH_HOLDOUT if is_holdout else GT_PATH

        # TODO set iters?
        if self.terminal:
            # iters = 100
            iters = 15000
        else:
            # iters = 50
            iters = 5000

        return train(train_keys_path, val_keys_path, save_dir, method, category,
              iters, model_file_dir, lr, image_dir, gt_path)

    def test_model(self, writer, writer_name, duration, is_holdout):
        category = self.category
        # test_prefix = '{}_episode_{:04d}_update_{:03d}_RL'.format(
        #     category, self.episode, self.update)

        test_keys_path = join(checkdir(join(self.classifier_root, 'loader')),
                              'test_keys.p')
        past_test_keys_path = join(self.classifier_root, 'past_test_keys.p')

        # Load past test keys
        if exists(past_test_keys_path):
            past_test_keys = pickle.load(open(past_test_keys_path, 'rb'))
        else:
            past_test_keys = []

        test_keys_curr = self.chosen_test['key']
        test_keys_all = test_keys_curr + past_test_keys

        pickle.dump(test_keys_all, open(test_keys_path, 'wb'))

        # Save past test keys for next time to use
        pickle.dump(test_keys_all, open(past_test_keys_path, 'wb'))

        save_dir = checkdir(join(self.classifier_root, 'snapshots'))
        os.makedirs(save_dir, exist_ok=True)
        method = 'resnet'
        image_dir = IMAGE_DIR_HOLDOUT if is_holdout else IMAGE_DIR_TRAIN
        gt_path = GT_PATH_HOLDOUT if is_holdout else GT_PATH

        return test(test_keys_path, save_dir, method, category, writer, writer_name,
                    duration, image_dir, gt_path)
