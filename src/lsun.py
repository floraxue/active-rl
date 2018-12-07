from util import logger, softmax
from env_obj import Env
import os
from os.path import join, exists
import numpy as np
import pickle
from collections import namedtuple
import subprocess
from dataset import FeatDataset, ImageData
from train_new import MACHINE_LABEL_DIR_HOLDOUT, IMAGE_DIR_HOLDOUT, GT_PATH_HOLDOUT

Sample = namedtuple('Sample', ('feat', 'label', 'key'))


def train_lsun_model(game, train_mode, work_root):
    load_keys(game, train_mode, work_root)

    category = 'cat'
    work_dir = join(work_root, train_mode)

    train_keys_path = join(work_dir, 'loader', 'train_keys.p')
    val_keys_path = join(work_dir, 'loader', 'val_keys.p')

    save_dir = join(work_dir, 'snapshots')
    model_file_dir = join(work_root, 'latest_RL', 'snapshots')
    os.makedirs(save_dir, exist_ok=True)
    method = 'resnet'

    # TODO set iters?
    if game.terminal:
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


def test_lsun_model(test_mode, work_root):
    category = 'cat'
    work_dir = join(work_root, test_mode)
    test_keys_path = join(work_dir, 'loader', 'test_keys.p')
    assert exists(test_keys_path)

    save_dir = join(work_dir, 'snapshots')
    os.makedirs(save_dir, exist_ok=True)
    method = 'resnet'

    cmd = 'python3 -m vfg.label.train_new test -e {0} -s {1} ' \
          '-m {2} --category {3}'.format(
            test_keys_path, save_dir, method, category)

    output = subprocess.check_output(cmd, shell=True,
                                     stderr=subprocess.STDOUT)

def test_lsun_model_holdout(test_mode, work_root):
    category = 'cat'
    work_dir = join(work_root, test_mode)
    test_keys_path = join(work_dir, 'loader', 'test_keys.p')
    assert exists(test_keys_path)

    save_dir = join(work_dir, 'snapshots')
    os.makedirs(save_dir, exist_ok=True)
    method = 'resnet'

    cmd = 'python3 -m vfg.label.train_new test -e {0} -s {1} ' \
          '-m {2} --category {3}'.format(
            test_keys_path, save_dir, method, category)

    output = subprocess.check_output(cmd, shell=True,
                                     stderr=subprocess.STDOUT)

#train once per iteration
def train_lsun_model_holdout(game, train_mode, work_root, new_key_path):
    load_keys_holdout(game, work_root, train_mode, new_key_path)

    category = 'cat'
    work_dir = join(work_root, train_mode)

    train_keys_path = join(work_dir, 'loader', 'train_keys.p')
    val_keys_path = join(work_dir, 'loader', 'val_keys.p')

    save_dir = join(work_dir, 'snapshots')
    model_file_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    method = 'resnet'

    # TODO set iters?
    if game.terminal:
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

def load_keys(game, train_mode, work_root):
    work_dir = join(work_root, train_mode)

    train_keys_path = join(work_dir, 'loader', 'train_keys.p')
    val_keys_path = join(work_dir, 'loader', 'val_keys.p')
    test_keys_path = join(work_dir, 'loader', 'test_keys.p')
    past_train_keys_path = join(work_root, 'past_train_keys.p')
    past_val_keys_path = join(work_root, 'past_val_keys.p')
    past_test_keys_path = join(work_root, 'past_test_keys.p')

    # Load past train val test keys
    if exists(past_train_keys_path):
        past_train_keys = pickle.load(open(past_train_keys_path, 'rb'))
    else:
        past_train_keys = []
    if exists(past_val_keys_path):
        past_val_keys = pickle.load(open(past_val_keys_path, 'rb'))
    else:
        past_val_keys = []
    if exists(past_test_keys_path):
        past_test_keys = pickle.load(open(past_test_keys_path, 'rb'))
    else:
        past_test_keys = []

    # Sample LSUN chosen set
    train_data= map(lambda x: Sample(*x), game.train_data[game.order[game.last: game.index]])
    assert game.latest_num_train and game.latest_num_val and game.latest_num_test
    chosen_set = np.random.choice(train_data, game.latest_curr_iter_num)

    train_keys = []
    train_labels = []
    val_keys = []
    val_labels = []
    test_keys = []
    test_labels = []

    for i, sample in enumerate(chosen_set):
        if i < game.latest_num_train:
            train_keys.append(sample.key)
            train_labels.append(sample.label)
        elif i < game.latest_num_train + game.latest_num_val:
            val_keys.append(sample.key)
            val_labels.append(sample.label)
        else:
            test_keys.append(sample.key)
            test_labels.append(sample.label)

    # Balance labels for current train val set
    train_order = game.balance_labels(train_labels)
    train_keys_curr = [train_keys[i] for i in train_order]
    val_bal_order = game.balance_labels(val_labels)
    val_bal_keys_curr = [val_keys[i] for i in val_bal_order]

    train_keys_all = train_keys_curr + past_train_keys
    val_keys_all = val_bal_keys_curr + past_val_keys
    test_keys_all = test_keys + past_test_keys

    pickle.dump(train_keys_all, open(train_keys_path, 'wb'))
    pickle.dump(val_keys_all, open(val_keys_path, 'wb'))
    pickle.dump(test_keys_all, open(test_keys_path, 'wb'))

#work_root: classifier_holdout, test_mode: latest_LSUN
def load_keys_holdout(game, train_mode, work_root, new_key_path):
    work_dir = join(work_root, train_mode)

    train_keys_path = join(work_dir, 'loader', 'train_keys.p')
    val_keys_path = join(work_dir, 'loader', 'val_keys.p')
    test_keys_path = join(work_dir, 'loader', 'test_keys.p')
    past_train_keys_path = join(work_root, 'past_train_keys_LSUN.p')
    past_val_keys_path = join(work_root, 'past_val_keys_LSUN.p')
    past_test_keys_path = join(work_root, 'past_test_keys_LSUN.p')

    # Load past train val test keys
    if exists(past_train_keys_path):
        past_train_keys = pickle.load(open(past_train_keys_path, 'rb'))
    else:
        past_train_keys = []
    if exists(past_val_keys_path):
        past_val_keys = pickle.load(open(past_val_keys_path, 'rb'))
    else:
        past_val_keys = []
    if exists(past_test_keys_path):
        past_test_keys = pickle.load(open(past_test_keys_path, 'rb'))
    else:
        past_test_keys = []

    # Sample LSUN chosen set

    dataset =  ImageData(key_path = new_key_path, image_dir = IMAGE_DIR_HOLDOUT, gt_path= GT_PATH_HOLDOUT)
    order = np.random.permutation(range(len(dataset)))

    chosen_set= map(lambda x: Sample(*x), dataset[order[:game.budget]])
    assert game.latest_num_train and game.latest_num_val and game.latest_num_test

    train_keys = []
    train_labels = []
    val_keys = []
    val_labels = []
    test_keys = []
    test_labels = []

    num_val = int(game.val_rate * game.budget)
    num_test = int(game.test_rate * game.budget)

    for i, sample in enumerate(chosen_set):
        if i < num_val:
            val_keys.append(sample.key)
            val_labels.append(sample.label)
        elif i < num_val + num_test:
            test_keys.append(sample.key)
            test_labels.append(sample.label)
        else:
            train_keys.append(sample.key)
            train_labels.append(sample.label)
    logger.info("LSUN: num train {0}, num val {1}, num test".format(len(train_keys), len(val_keys), len(test_keys)))

    # Balance labels for current train val set
    train_order = game.balance_labels(train_labels)
    train_keys_curr = [train_keys[i] for i in train_order]
    val_bal_order = game.balance_labels(val_labels)
    val_bal_keys_curr = [val_keys[i] for i in val_bal_order]

    train_keys_all = train_keys_curr + past_train_keys
    val_keys_all = val_bal_keys_curr + past_val_keys
    test_keys_all = test_keys + past_test_keys

    pickle.dump(train_keys_all, open(train_keys_path, 'wb'))
    pickle.dump(val_keys_all, open(val_keys_path, 'wb'))
    pickle.dump(test_keys_all, open(test_keys_path, 'wb'))

    # Save past keys for next time to use
    pickle.dump(train_keys_all, open(past_train_keys_path, 'wb'))
    pickle.dump(val_keys_all, open(past_val_keys_path, 'wb'))
    pickle.dump(test_keys_all, open(past_test_keys_path, 'wb'))