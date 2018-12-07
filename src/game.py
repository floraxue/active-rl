import torch
import torch.nn as nn
import numpy as np
import sys
import random
import subprocess
from collections import namedtuple
import os
from os.path import join, exists
from util import logger
from dataset import FeatDataset
import pickle

Sample = namedtuple('Sample', ('feat', 'label', 'key'))


class VFGGAME:

    def __init__(self, args):
        self.category = args.category
        self.key_path = args.key_path
        self.feat_dir = args.feat_dir
        self.gt_path = args.gt_path
        self.train_data = FeatDataset(key_path=self.key_path,
                                      feat_dir=self.feat_dir,
                                      gt_path=self.gt_path)

        # decide the querying order
        self.order = np.random.permutation(range(len(self.train_data)))
        # header of the training set
        self.index = 0

        # # use the val data to get reward
        # self.val_data = FeatsData(
        #     out_dir=env.category_split_dir(category), prefix='val',
        #     category=category)
        #
        # # final testing when playing the game
        # self.test_data = FeatsData(
        #     out_dir=env.category_split_dir(category), prefix='test',
        #     category=category)

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
        self.last = 0

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
        self.order = np.random.permutation(range(len(self.train_data)))
        self.index = 0
        self.current_sample = None
        self.current_reward = 0

    def sample(self):
        """ return the feature of the current sample """
        item = self.train_data[self.order[self.index]]
        self.current_sample = Sample(*item)
        return self.current_sample.feat

    def step(self, action):
        if self.chosen == self.budget:
            self.terminal = True

        if action > 0:
            # take the current sample
            self.chosen += 1
            self.chosen_set['key'] += [self.current_sample.key]
            self.chosen_set['gt'] += [self.current_sample.label]

            if self.chosen % self.duration == 0 or self.chosen == self.budget:
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

                #recording this for LSUN
                self.latest_num_train = num_train
                self.latest_num_val = num_val
                self.latest_num_test = num_test
                self.update += 1

        # move to the next sample in the queue
        self.index += 1
        next_state = self.sample()

        return self.current_reward, next_state, self.terminal

    def train_model(self, train_mode, work_root):
        category = self.category
        # train_prefix = 'RL_{}_episode_{:04d}_update_{:03d}'.format(
        #     category, self.episode, self.update)
        work_dir = join(work_root, train_mode)

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
        train_keys = self.chosen_train['key']
        train_labels = self.chosen_train['gt']
        val_keys = self.chosen_val['key']
        val_labels = self.chosen_val['gt']

        train_order = self.balance_labels(train_labels)
        train_keys_curr = [train_keys[i] for i in train_order]
        val_bal_order = self.balance_labels(val_labels)
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
        model_file_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        method = 'resnet'

        # TODO set iters?
        if self.terminal:
            # iters = 15000
            iters = 15
        else:
            # iters = 5000
            iters = 5

        cmd = 'python3 -m vfg.label.train_new train -t {0} -e {1} -s {2} ' \
              '-m {3} --category {4} --iters {5} --model-file-dir {6}'.format(
                train_keys_path, val_keys_path, save_dir, method, category, iters,
                model_file_dir)

        output = subprocess.check_output(cmd, shell=True,
                                         stderr=subprocess.STDOUT)

    def balance_labels(self, labels):
        pos_indices = np.nonzero(labels > 0)[0]
        neg_indices = np.nonzero(labels <= 0)[0]
        num_pos = pos_indices.size
        num_neg = neg_indices.size
        num_half = max(num_pos, num_neg)
        logger.info('pos: %d neg: %d half: %d', num_pos, num_neg, num_half)
        if num_half > num_pos:
            pos_indices = np.concatenate([pos_indices for _ in
                                          range(int(num_half / num_pos + 1))],
                                         axis=0)
            pos_indices = pos_indices[:num_half]
        if num_half > num_neg:
            neg_indices = np.concatenate([neg_indices for _ in
                                          range(int(num_half / num_neg + 1))],
                                         axis=0)
            neg_indices = neg_indices[:num_half]
        new_indices = np.concatenate([pos_indices, neg_indices])
        order = np.random.permutation(num_half * 2)
        new_indices = new_indices[order]
        return new_indices

    def test_model(self, test_mode, work_root):
        category = self.category
        # test_prefix = '{}_episode_{:04d}_update_{:03d}_RL'.format(
        #     category, self.episode, self.update)
        work_dir = join(work_root, test_mode)

        test_keys_path = join(work_dir, 'loader', 'test_keys.p')
        past_test_keys_path = join(work_root, 'past_test_keys.p')

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

        save_dir = join(work_dir, 'snapshots')
        os.makedirs(save_dir, exist_ok=True)
        method = 'resnet'

        cmd = 'python3 -m vfg.label.train_new test -e {0} -s {1} ' \
              '-m {2} --category {3}'.format(
                test_keys_path, save_dir, method, category)

        output = subprocess.check_output(cmd, shell=True,
                                         stderr=subprocess.STDOUT)

    # def write_list(self):
    #     """dumping the training list to file"""
    #     episode = self.episode
    #     update = self.update
    #
    #     key_path= env.rl_sample_episode_key_path(
    #         self.category, episode, update)
    #     db_path = env.rl_sample_episode_db_path(
    #         self.category, episode, update)
    #     gt_path = env.rl_sample_episode_gt_path(
    #         self.category, episode, update)
    #
    #     with open(key_path, 'w') as key_file:
    #         with open(db_path, 'w') as db_file:
    #             with open(gt_path, 'w') as gt_file:
    #                 for k, d, g in zip(self.chosen_set['key'],
    #                                    self.chosen_set['db'],
    #                                    self.chosen_set['gt']):
    #                     print(k, file=key_file)
    #                     print(d, file=db_file)
    #                     print(k+' '+str(g), file=gt_file)
    #
    #     logger.info('write training file for update {} of '
    #                 'episode {} to {}'.format(update, episode, key_path))

