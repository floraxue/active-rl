#! /usr/bin/env python3

from __future__ import print_function, division
import os
import platform
from collections import Iterable
from datetime import datetime

from os.path import join, expanduser, split, dirname, exists
from util import logger, checkdir


class Env(object):
    def __init__(self, work_dir=None, test=False, sandbox=True):
        self.work_dir = work_dir
        self._web_work_dir = '/vfgdata'
        self.test = test
        self.part = None
        self.sandbox = sandbox
        if self.work_dir is None:
            try:
                self.work_dir = os.environ['VFGDIR']
                logger.info('Using VFGDIR {} defined in shell.'.format(
                    self.work_dir))
            except KeyError:
                self.work_dir = './vfgdata'
                logger.info('No work_dir is not set. Use {} instead.'.format(
                    self.work_dir))
        else:
            logger.info('Use work_dir: {}'.format(self.work_dir))
        if self.test:
            self.work_dir = join(self.work_dir, 'test')
        try:
            self.rname = os.environ['EXPNAME']
            logger.info('Using EXPNAME {} defined in shell'.format(self.rname))
        except KeyError:
            self.rname = 'new_iter'
            logger.info('No EXPNAME is defined. Use default {}.'.format(
                self.rname))
        try:
            self.remote_host_name = os.environ['REMOTE']
            logger.info('Using REMOTE {} defined in shell'.format(
                self.remote_host_name))
        except KeyError:
            self.remote_host_name = 'floraxue@aspidochelone.ist.berkeley.edu'
            logger.info('No REMOTE is defined. Use {} instead'.
                        format(self.remote_host_name))
        checkdir(self.work_dir)

    def workspace(self):
        return checkdir(self.work_dir)

    @staticmethod
    def mode():
        try:
            m = os.environ['MODE']
            logger.info('running with mode {}'.format(m))
        except KeyError:
            m = 'ITER'
            logger.info('No MODE is defined. Use default mode {}'.format(m))
        return m

    def remote_host(self):
        return self.remote_host_name

    def run_name(self):
        return self.rname

    @staticmethod
    def num_samples():
        try:
            n = int(os.environ['NUM_SAMPLES'])
            logger.info('Using NUM_SAMPLES {} defined in shell.'.format(n))
        except KeyError:
            n = 40000
            logger.info('NO NUM_SAMPLES is defined. '
                        'Use default value {}'.format(n))
        return n

    @staticmethod
    def num_val_samples():
        try:
            n = int(os.environ['NUM_VAL_SAMPLES'])
            logger.info('Using NUM_VAL_SAMPLES {} defined in shell.'.format(n))
        except KeyError:
            n = 10000
            logger.info('No NUM_VAL_SAMPLES defined. '
                        'Use default value {}'.format(n))
        return n

    @staticmethod
    def num_test_samples():
        try:
            n = int(os.environ['NUM_TEST_SAMPLES'])
            logger.info('Using NUM_TEST_SAMPLES {} '
                        'defined in shell.'.format(n))
        except KeyError:
            n = 0
            logger.info('No NUM_TEST_SAMPLES defined.'
                        'Use default value {}'.format(n))
        return n

    @staticmethod
    def num_test_all_samples():
        try:
            n = int(os.environ['NUM_TEST_ALL'])
            logger.info('Using NUM_TEST_ALL {} defined in shell'.format(n))
        except KeyError:
            n = 100000
            logger.info('No NUM_TEST_ALL defined. '
                        'Use default value {}'.format(n))
        return n

    @staticmethod
    def num_stop():
        try:
            n = int(os.environ['NUM_STOP'])
            logger.info('Using NUM_STOP {} defined in shell.'.format(n))
        except KeyError:
            n = 5000
            logger.info('No NUM_STOP defined. '
                        'Use default value {}'.format(n))
        return n

    def log_dir(self):
        return checkdir(join(self.workspace(), 'log', self.run_name()))
        # return checkdir('/scratch/fy/vfgdata/log', self.run_name())

    def tensorboard_log_dir(self, folder):
        time_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
        return join(self.log_dir(), 'tb', time_str, folder)

    def json_log_dir(self):
        try:
            log_name = os.environ['LOGNAME']
            log_path = join(self.log_dir(), 'json', log_name)
            logger.info("Using log_path {}".format(log_path))
        except KeyError:
            log_path = '/data3/floraxue/tmp/log/json'
            logger.info("Default log_path {}".format(log_path))
        return log_path

    def data_dir(self):
        return self.workspace() + '/data'

    def labeling_dir(self):
        return checkdir(join(self.workspace(), 'labeling', self.run_name()))

    def split_dir(self):
        return checkdir(join(self.workspace(), 'data_split'))

    def category_split_dir(self, category):
        return checkdir(join(self.split_dir(), category))

    def train_split_key_path(self, category):
        return join(self.category_split_dir(category), 'train.txt')

    def train_split_db_path(self, category):
        return join(self.category_split_dir(category), 'train_db.txt')

    def train_split_label_path(self, category):
        return join(self.category_split_dir(category), 'train_gt.txt')

    def test_split_key_path(self, category):
        return join(self.category_split_dir(category), 'test.txt')

    def test_split_db_path(self, category):
        return join(self.category_split_dir(category), 'test_db.txt')

    def test_split_label_path(self, category):
        return join(self.category_split_dir(category), 'test_gt.txt')

    def val_split_key_path(self, category):
        return join(self.category_split_dir(category), 'val.txt')

    def val_split_db_path(self, category):
        return join(self.category_split_dir(category), 'val_db.txt')

    def val_split_label_path(self, category):
        return join(self.category_split_dir(category), 'val_gt.txt')

    def init_key_path(self, category):
        try:
            key_file = join(self.trial_dir(category),
                            os.environ['init_key_file'])
        except KeyError:
            key_file = self.train_split_key_path(category)
        return key_file

    def init_db_path(self, category):
        try:
            db_file = join(self.trial_dir(category),
                           os.environ['init_db_file'])
        except KeyError:
            db_file = self.train_split_db_path(category)
        return db_file

    def init_label_path(self, category):
        try:
            lab_file = join(self.trial_dir(category),
                            os.environ['init_label_file'])
        except KeyError:
            lab_file = self.train_split_label_path(category)
        return lab_file

    def trial_dir(self, category):
        return checkdir(join(self.labeling_dir(), category, 'trials'))

    # def trial_path_prefix(self, category, trial):
    #     return join(self.sample_label_root(), category, 'trials',
    #                 '{0}_trial_{1:03d}'.format(category, trial))

    def trial_key_path(self, category, trial):
        return join(self.trial_dir(category),
                    '{0}_trial_{1:04d}.txt'.format(category, trial))

    def trial_db_path(self, category, trial):
        return join(self.trial_dir(category),
                    '{0}_trial_{1:04d}_db.txt'.format(category, trial))

    def trial_turk_path(self, category, trial):
        return join(self.trial_dir(category),
                    '{0}_trial_{1:04d}_turk.txt'.format(category, trial))

    def labels_root(self, category):
        return checkdir(join(self.labeling_dir(), category, 'labels'))

    def category_data_dir(self, category):
        return self.download_data_dir(category)

    def category_raw_data_path(self, category, index):
        return join(self.category_data_dir(category),
                    '{0}_{1:04d}_lmdb'.format(category, index))

    def category_webp_256_stage_1_1_data_path(self, category, part):
        # return stage_export_data_path(category, 1, 1, part, 'webp', 256)
        # xin: redefine the data path for the offline benchmark
        return self.category_raw_data_path(category, part)

    def category_archive_data_dir(self, category):
        return join(self.data_dir(), 'archive', category)

    def category_archive_data_path(self, category, index):
        return join(self.category_archive_data_dir(category),
                    '{0}_{1:04d}_lmdb'.format(category, index))

    def download_data_dir(self, category):
        return checkdir(join(self.workspace(), 'data', 'raw', category))

    def download_log_dir(self, category):
        return checkdir(
            join(self.log_dir(), 'download', category))

    def cmd_dir(self):
        return join(self.workspace(), 'cmds')

    def category_cmd_dir(self, category):
        return join(self.cmd_dir(), category)

    def list_root(self):
        return join(self.workspace(), 'lists')

    def category_list_root(self, category):
        return join(self.list_root(), category)

    def category_list_dir(self, category):
        return join(self.list_root(), category)

    def feat_dir(self):
        return join(self.workspace(), 'feats')

    def category_feat_dir(self, category):
        return join(self.feat_dir(), category)

        if type(stage) == str:
            path = join(self.labels_root(category),
                        '{}_{}'.format(category, stage))
        else:
            path = join(self.labels_root(category),
                        '{}_{:02d}'.format(category, stage))
        if label is not None:
            path += '_{}'.format(label)
        return path

    def label_result_key_path(self, category, stage, trials, label, prefix=''):
        if prefix:
            return self.label_result_prefix(category, stage, trials, label) \
                   + '_{}.txt'.format(prefix)
        else:
            return self.label_result_prefix(category, stage, trials, label) \
                   + '.txt'

    def label_result_db_path(self, category, stage, trials, label, prefix=''):
        if prefix:
            return self.label_result_prefix(category, stage, trials, label) + \
                   '_{}_db.txt'.format(prefix)
        else:
            return self.label_result_prefix(category, stage, trials, label) + \
                   '_db.txt'

    def label_result_gt_path(self, category, stage, trials, label, prefix=''):
        if prefix:
            return self.label_result_prefix(category, stage, trials, label) \
                   + '_{}_gt.txt'.format(prefix)
        else:
            return self.label_result_prefix(category, stage, trials, label) + \
                   '_gt.txt'

    def label_result_info_path(self, category, stage, trials, prefix=''):
        if prefix:
            return self.label_result_prefix(category, stage, trials) \
                   + '_{}_info.txt'.format(prefix)
        else:
            return self.label_result_prefix(category, stage, trials) \
                   + '_info.txt'

    def label_path(self, category, trial, label=0):
        return self.label_path_prefix(category, trial, label) \
               + '.txt'

    def label_path_prefix(self, category, trial, label=0):
        return join(self.labels_root(category),
                    '{0}_trial_{1:03d}_label_{2:02d}'.format(
                        category, trial, label))

    def sample_root(self):
        # return workspace() + '/fy/samples'
        return join(self.workspace(), 'samples')

    def sample_label_root(self):
        return self.labeling_dir()

    def make_trial_name(self, trials):
        if not isinstance(trials, Iterable):
            trials = [trials]
        return '_'.join(['{:02d}'.format(t) for t in trials])

    def make_mlp_dir(self, category, trials):
        return self.make_split_work_dir(category, trials, 'mlp')

    def labeling_split_dir(self, category):
        return join(self.labeling_dir(), category, 'split')

    def make_split_work_dir(self, category, trials, method):
        if not isinstance(trials, Iterable):
            trials = [trials]
        return join(self.labeling_split_dir(category),
                    '{}_{}_{}'.format(category, self.make_trial_name(trials),
                                      method))

    @staticmethod
    def mturk_list_dir():
        return join(dirname(__file__), '../../mturk')

    def mturk_data_dir(self):
        return checkdir(join(self.workspace(), 'mturk'))

    def mturk_result_dir(self):
        return checkdir(join(self.mturk_data_dir(), 'results'))

    def mturk_www_dir(self, subdir=''):
        assert subdir in ['', 'results', 'hits', 'assignments', 'submissions',
                          'jobs'], \
            'If you feel necessary to create a new subfolder for mturk, ' \
            'please also edit this line'
        return checkdir(join(self.workspace(), 'mturk_www',
                             'sandbox' if self.sandbox else 'release',
                             self.run_name(), subdir))

    def mturk_decision_path(self):
        return join(checkdir(join(self.mturk_www_dir(), 'submissions')),
                    'decisions.db')

    def mturk_host(self):
        if self.sandbox:
            return 'mechanicalturk.sandbox.amazonaws.com'
        else:
            return 'mechanicalturk.amazonaws.com'

    def web_host(self):
        return 'visual.berkeley.edu'

    def web_pem(self):
        return 'visual_berkeley_edu_aws_key.pem'

    def web_workspace(self):
        return self._web_work_dir

    def web_mturk_dir(self, subdir=''):
        assert subdir in ['', 'results', 'hits', 'assignments', 'submissions',
                          'jobs'], 'Web Mturk Workspace Dir'
        dir_name = join(self.web_workspace(), 'mturk_www',
                        'sandbox' if self.sandbox else 'release',
                        self.run_name(), subdir)
        if not exists(dir_name):
            os.makedirs(dir_name)
        return dir_name

    def data_turk_dir(self):
        return checkdir(join(self.data_dir(), 'turk', self.run_name()))

    def data_raw_dir(self, category):
        return checkdir(join(self.data_dir(), 'raw', category))

    def image_data_turk_path(self, category, key):
        # image_data_turk_dir = checkdir(join(self.data_turk_dir(category),
        #                                     '/'.join(key[:6])))
        image_data_turk_dir = checkdir(
            join(self.data_turk_dir(), category, key[:4]))
        return join(image_data_turk_dir, key + '.jpg')

    def category_data_turk_dir(self, category):
        return checkdir(join(self.data_turk_dir(), category))

    def image_data_turk_link(self, category, key, turk=False):
        # url_root = 'https://s3.us-east-2.amazonaws.com/bdd-mturk-data'
        # return join(url_root, category, '/'.join(key[:6]), key + '.jpg')
        url_root = generate_aws_root(turk)
        return join(url_root, category, '/'.join(key[:4]), key + '.jpg')

    def robottxt_path(self):
        # return checkdir(join(self.workspace(), 'data', 'robottxt',
        #                     'robottxt_{}.sqlite'.format(platform.node())))
        if self.part is None:
            return checkdir(join(self.workspace(), 'data', 'robottxt',
                                 'robottxt_{}_db'.format(platform.node())))
        else:
            return checkdir(join(
                self.workspace(), 'data', 'robottxt',
                'robottxt_{0}_{1}_db'.format(platform.node(), self.part)))

    def gpu_lock_dir(self):
        return join(self.workspace(), 'gpu_lock')

    # Xin: use this path to determine the algorithm labeling results
    def label_dir(self, category):
        return checkdir(join(self.workspace(), 'machine_labels',
                             self.run_name(), category))

    def label_result_prefix(self, category, stage, trials, label=None):
        trial_prefix = 'trials_'+'_'.join(['{:03d}'.format(t) for t in trials])
        if type(stage) == str:
            path = join(self.label_dir(category), '{}_{}_stage_{}'.format(
                category, trial_prefix, stage))
        else:
            path = join(self.label_dir(category),
                        '{}_{}_stage_{:02d}'.format(category,
                                                    trial_prefix, stage))
        if label is not None:
            path += '_{}'.format(label)
        return path

    def gt_dir(self, category):
        return join(self.workspace(), 'ground_truth', category)

    def index_dir(self, category):
        return join(self.data_dir(), 'dump_index', category)

    def index_key_path(self, category, trial):
        return join(self.index_dir(category),
                    '{0}_lmdb_{1:04d}.txt'.format(category, trial))

    def index_db_path(self, category, trial):
        return join(self.index_dir(category),
                    '{0}_lmdb_{1:04d}_db.txt'.format(category, trial))
