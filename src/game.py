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

Sample = namedtuple('Sample', ('feat', 'label', 'key'))


class VFGGAME:

    def __init__(self, args):
        self.category = category = args.category
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
        self.chosen_set = {
            'key': [], 'gt': []}
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
        self.chosen_train = {
            'key': [], 'gt': []}
        self.chosen_val = {
            'key': [], 'gt': []}
        self.chosen_test = {
            'key': [], 'gt': []}

    def reset(self, new_key_path):
        self.chosen = 0
        self.chosen_set = {
            'key': [], 'gt': []}
        self.chosen_train = {
            'key': [], 'gt': []}
        self.chosen_val = {
            'key': [], 'gt': []}
        self.chosen_test = {
            'key': [], 'gt': []}
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
                self.chosen_train['key'] = self.chosen_set['key'][:num_train]
                self.chosen_train['gt'] = self.chosen_set['gt'][:num_train]
                self.chosen_val['key'] = self.chosen_set['key'][num_train: num_train + num_val]
                self.chosen_val['gt'] = self.chosen_set['gt'][num_train: num_train + num_val]
                self.chosen_test['key'] = self.chosen_set['key'][-num_test:]
                self.chosen_test['gt'] = self.chosen_set['gt'][-num_test:]
                # self.update += 1
                # self.train_model()

        # move to the next sample in the queue
        self.index += 1
        next_state = self.sample()

        return self.current_reward, next_state, self.terminal

    # def train_model(self):
    #     self.write_list()
    #     category = self.category
    #     train_dir = env.category_rl_dir(category)
    #     train_prefix = '{}_episode_{:04d}_update_{:03d}'.format(
    #         category, self.episode, self.update)
    #
    #     save_dir = join(train_dir, 'snapshots')
    #     os.makedirs(save_dir, exist_ok=True)
    #     eval_dir = env.category_split_dir(category)
    #     method = 'resnet'
    #
    #     if self.terminal:
    #         iters = 15000
    #     else:
    #         iters = 5000
    #
    #     cmd = 'python3 -m vfg.label.train_new trian -t {0} -e {1} -s {2} ' \
    #           '-m {3} --category {4} --iters {5} --train-prefix {6}'.format(
    #             train_dir, eval_dir, save_dir, method, category, iters,
    #             train_prefix)
    #
    #     output = subprocess.check_output(cmd, shell=True,
    #                                      stderr=subprocess.STDOUT)
    #
    #     self.current_reward = float(output.decode('utf-8'))
    #     logger.info('current reward in update {} of episode {} is {}'.format(
    #         self.update, self.episode, self.current_reward))
    #
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

