import numpy as np
import subprocess
from collections import namedtuple
import os
from os.path import join, exists
from train_new import IMAGE_DIR_TRAIN, CLASSIFIER_ROOT
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
from train_new import MACHINE_LABEL_DIR, USE_PRETRAINED

Sample = namedtuple('Sample', ('feat', 'label', 'key'))

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class VFGGAME:

    def __init__(self, args):
        self.category = args.category
        self.key_path = args.key_path
        self.feat_dir = args.feat_dir
        self.gt_path = args.gt_path
        self.train_data = FeatDataset(key_path=self.key_path,
                                      feat_dir=self.feat_dir,
                                      gt_path=self.gt_path)

        self.image_data = ImageData(key_path = self.key_path,
                                    image_dir = IMAGE_DIR_TRAIN,
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
        curr_model_path = join(CLASSIFIER_ROOT, "latest_RL", "snapshots", 'bal_model_best.pth.tar')
        model = models.resnet18(pretrained=False, num_classes = 2)
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

    def train_model(self, train_mode, work_root, lr):
        category = self.category
        # train_prefix = 'RL_{}_episode_{:04d}_update_{:03d}'.format(
        #     category, self.episode, self.update)
        work_dir = checkdir(join(work_root, train_mode))

        train_keys_path = join(checkdir(join(work_dir, 'loader')), 'train_keys.p')
        val_keys_path = join(checkdir(join(work_dir, 'loader')), 'val_keys.p')
        past_train_keys_path = join(work_root, 'past_train_keys.p')
        past_val_keys_path = join(work_root, 'past_val_keys.p')

        # Load past train val keys
        if exists(past_train_keys_path):
            past_train_keys = pickle.load(open(past_train_keys_path, 'rb'))
        else:
            initial_train_keys_path = join('/data3/floraxue/cs294/active-rl-data', 'initial_trial_keys.p')
            # initial_train_keys_path = join('/data3/floraxue/cs294/active-rl-data', 'initial_trial_keys_holdout.p')
            past_train_keys = pickle.load(open(initial_train_keys_path, 'rb'))

        if exists(past_val_keys_path):
            past_val_keys = pickle.load(open(past_val_keys_path, 'rb'))
        else:
            past_val_keys = []

        # Balance labels for current train val set
        train_keys = self.chosen_train['key']
        train_labels = self.chosen_train['gt']
        val_keys = self.chosen_val['key']
        val_labels = self.chosen_val['gt']

        # train_order = self.balance_labels(train_labels)
        # train_keys_curr = [train_keys[i] for i in train_order]
        # val_bal_order = self.balance_labels(val_labels)
        # val_bal_keys_curr = [val_keys[i] for i in val_bal_order]
        train_keys_curr = train_keys
        val_bal_keys_curr = val_keys

        # TODO: assume we don't need val keys before bal
        train_keys_all = train_keys_curr + past_train_keys
        val_keys_all = val_bal_keys_curr + past_val_keys

        # Save past train val keys for next time to use
        pickle.dump(train_keys_all, open(past_train_keys_path, 'wb'))
        pickle.dump(val_keys_all, open(past_val_keys_path, 'wb'))

        #use only a portion of the past_train_keys for this time
        past_train_keys_curr = np.random.choice(past_train_keys, size=len(past_train_keys)//5, replace = False).tolist()
        pickle.dump(train_keys_curr + past_train_keys_curr, open(train_keys_path, 'wb'))
        pickle.dump(val_keys_all, open(val_keys_path, 'wb'))

        save_dir = checkdir(join(work_dir, 'snapshots'))
        model_file_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        method = 'resnet'

        # TODO set iters?
        if self.terminal:
            # iters = 100
            iters = 15000
        else:
            # iters = 50
            iters = 5000

        return train(train_keys_path, val_keys_path, save_dir, method, category,
              iters, model_file_dir, lr)

    def balance_labels(self, labels):
        pos_indices = np.nonzero(list(map(lambda x: x+1,labels)))[0]
        neg_indices = np.nonzero(list(map(lambda x: x-1,labels)))[0]
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

    def test_model(self, test_mode, work_root, writer, name, duration):
        category = self.category
        # test_prefix = '{}_episode_{:04d}_update_{:03d}_RL'.format(
        #     category, self.episode, self.update)
        work_dir = checkdir(join(work_root, test_mode))

        test_keys_path = join(checkdir(join(work_dir, 'loader')), 'test_keys.p')
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

        save_dir = checkdir(join(work_dir, 'snapshots'))
        os.makedirs(save_dir, exist_ok=True)
        method = 'resnet'

        return test(test_keys_path, save_dir, method, category, writer, name, duration)

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

