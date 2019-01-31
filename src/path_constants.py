import os

# Static folders

# Image dir
IMAGE_DIR_TRAIN = '/data3/floraxue/cs294/active-rl-data/data/images/train/cat'
IMAGE_DIR_FIXED = '/data3/floraxue/cs294/active-rl-data/data/images/fixed/cat'
IMAGE_DIR_HOLDOUT = '/data3/floraxue/cs294/active-rl-data/data/images/holdout/cat'
# Feature dir
FEAT_DIR_TRAIN = "/data3/floraxue/cs294/active-rl-data/data/feats/train/cat"
FEAT_DIR_HOLDOUT = "/data3/floraxue/cs294/active-rl-data/data/feats/holdout/cat"
# Ground truth
GT_PATH = '/data3/floraxue/cs294/active-rl-data/ground_truth/cat_gt_cached.p'
GT_PATH_HOLDOUT = '/data3/floraxue/cs294/active-rl-data/ground_truth/cat_gt_cached_holdout.p'
# Initial dataset keys
UNSURE_KEY_PATH = '/data3/floraxue/cs294/active-rl-data/cat_trial_0_unsure.p'
UNSURE_KEY_PATH_HOLDOUT = '/data3/floraxue/cs294/active-rl-data/cat_trial_0_unsure_holdout.p'
# Initial trial keys for the first train
INITIAL_TRIAL_KEY_PATH = '/data3/floraxue/cs294/active-rl-data/initial_trial_keys.p'
INITIAL_TRIAL_KEY_PATH_HOLDOUT = '/data3/floraxue/cs294/active-rl-data/initial_trial_keys_holdout.p'

# Experiment folders
assert os.environ['EXPNAME']
exp = os.environ['EXPNAME']
MACHINE_LABEL_DIR = '/data3/floraxue/cs294/exp/{0}/machine_labels'.format(exp)
MACHINE_LABEL_DIR_HOLDOUT = '/data3/floraxue/cs294/exp/{0}/machine_labels_holdout'.format(exp)
CLASSIFIER_ROOT = '/data3/floraxue/cs294/exp/{0}/classifier'.format(exp)
CLASSIFIER_ROOT_HOLDOUT = '/data3/floraxue/cs294/exp/{0}/classifier_holdout'.format(exp)

# Resnet Model Url
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}