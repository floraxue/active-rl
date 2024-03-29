import sys
import torch
import torch.optim as optim

from itertools import count
import argparse
from os.path import join
import pickle

from agent import NSQ
from policy import NewPolicyNet

from buffer import ReplayMemory
from game import VFGGAME
from explorer import Explorer
from util import logger
from train_new import test_fixed_set, test_all
import random
from tensorboardX import SummaryWriter
from path_constants import *

writer = SummaryWriter('runs-rl-saveagent/')
# exp = 'rl-train'

def parse_arguments():
    parser = argparse.ArgumentParser(description="training N-step Q learning")
    parser.add_argument('--category', type=str, default='cat',
                        help='image category')
    parser.add_argument('--budget', type=int, default=1000,
                        help='maximum number of examples for human annotation')
    parser.add_argument('--eps-start', type=float, default=0.9,
                        help='starting epsilon')
    parser.add_argument('--eps-end', type=float, default=0.05,
                        help='ending epsilon')
    parser.add_argument('--decay-steps', type=int, default=100000,
                        help='decay steps')
    parser.add_argument('--gamma', type=float, default=0.999,
                        help='discount factor')
    parser.add_argument('--duration', '-k', type=int, default=25,
                        help='get reward every k steps')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='batch size')
    parser.add_argument('--target-update', '-T', type=int, default=1000,
                        help='update target network every T steps')
    parser.add_argument('--learning-start', type=int, default=50000)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--num-actions', type=int, default=2,
                        help='default action is `keep` or `drop`')
    parser.add_argument('--input_dim', type=int, default=2050,
                        help='feature size')
    parser.add_argument('--save-every', type=int, default=1,
                        help='save the checkpoint every K episode')
    parser.add_argument('--episodes', type=int, default=20)

    # flags for the game
    parser.add_argument('--eval-dir', type=str, default='',
                        help='path to the training list folder')
    parser.add_argument('--train-prefix', type=str, default='train',
                        help='prefix of the training files')
    # parser.add_argument('--feat-dir', type=str,
    #                     default='/data3/floraxue/cs294/active-rl-data/data/feats/train/cat')
    # parser.add_argument('--gt-path', type=str)
    # parser.add_argument('--image-dir', type=str)
    parser.add_argument('--val_rate', type=float, default=0.2)
    parser.add_argument('--test_rate', type=float, default=0.2)
    parser.add_argument('--mode', type=str, default='rl',
                        help='Sampling policy. '
                             'Should be one of rl, random or uncertainty')
    # parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='')

    args = parser.parse_args()
    # global exp
    # exp = args.exp
    # logger.info("Running exp {}".format(exp))
    if args.pretrained == '' and args.mode == 'rl':
        # not pretrained, so use train set
        args.key_path = UNSURE_KEY_PATH
        args.feat_dir = FEAT_DIR_TRAIN
        args.gt_path = GT_PATH
        args.image_dir = IMAGE_DIR_TRAIN
        args.initial_train_keys_path = INITIAL_TRIAL_KEY_PATH
        args.classifier_root = CLASSIFIER_ROOT
    elif args.pretrained != '':
        args.key_path = UNSURE_KEY_PATH_HOLDOUT
        args.feat_dir = FEAT_DIR_HOLDOUT
        args.gt_path = GT_PATH_HOLDOUT
        args.image_dir = IMAGE_DIR_HOLDOUT
        args.initial_train_keys_path = INITIAL_TRIAL_KEY_PATH_HOLDOUT
        args.classifier_root = CLASSIFIER_ROOT_HOLDOUT

    return args


def calculate_reward(category, i_episode, update):
    """
    Calculate reward from LSUN and RL
    :return:
    """
    prefix = '{}_episode_{:04d}_update_{:03d}'.format(category, i_episode, update)
    rl_file = join(CLASSIFIER_ROOT, 'fixed_set_acc_RL_{}.p'.format(prefix))
    rl_acc = pickle.load(open(rl_file, 'rb'))
    return rl_acc


def fixed_set_evaluation(category, mode, i_episode, update):
    """
    Get test accuracy on a fixed set to calculate reward
    :param category:
    :param mode:
    :param i_episode:
    :param update:
    :return:
    """
    test_keys_path = '/data3/floraxue/cs294/active-rl-data/pool/{}_fixed_keys.p'.format(category)
    method = 'resnet'
    model_file_dir = join(CLASSIFIER_ROOT, 'snapshots')
    test_prefix = '{}_{}_episode_{:04d}_update_{:03d}'.format(mode, category, i_episode, update)
    return test_fixed_set(test_keys_path, method, category, test_prefix, model_file_dir)


def test_all_data(category, i_episode, last_trial_key_path, is_holdout):
    """
    test to split the dataset
    :return:
    """
    trial = i_episode
    model_file_dir = join(CLASSIFIER_ROOT, 'snapshots')
    image_dir = IMAGE_DIR_HOLDOUT if is_holdout else IMAGE_DIR_TRAIN
    gt_path = GT_PATH_HOLDOUT if is_holdout else GT_PATH
    test_all(last_trial_key_path, trial, 'resnet', category, model_file_dir,
             image_dir, gt_path)


def run_pipeline(args, game):
    # build model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = args.input_dim
    num_actions = args.num_actions
    is_holdout = (args.pretrained != '')

    if args.mode == 'rl':
        memory = ReplayMemory(args.buffer_size)
        q_func = NewPolicyNet
        Q = q_func(input_dim, num_actions)
        if args.pretrained != '':
            Q.load_state_dict(torch.load(args.pretrained))
        Q = Q.to(device)
        target_Q = q_func(input_dim, num_actions).to(device)

        optimizer = optim.RMSprop(Q.parameters())

        expr = Explorer(args.eps_start, args.eps_end, decay_steps=args.decay_steps)

        robot = NSQ(Q, target_Q, optimizer, expr,
                    gamma=args.gamma, num_actions=num_actions)

    episode_durations = []

    # Pipeline params
    category = args.category
    # Set initial unsure key path
    new_key_path = args.key_path

    for i_episode in range(1, args.episodes + 1):
        lr = adjust_learning_rate(i_episode)
        game.reset(new_key_path)

        if len(game.train_data) < game.budget:
            logger.info("Unsure set is smaller than budget size. End this experiment.")
            break

        # pipeline param
        trial = i_episode

        # this is to reset the hidden units, not repackage the variables
        # robot.q_function.reset_hidden()
        # robot.target_q_function.reset_hidden()

        # sample the initial feature from the environment
        # since our policy network takes the hidden state and the current
        # feature as input. The hidden state is passed implicitly
        state = game.sample()
        state_seq, act_seq, reward_seq = [], [], []
        next_state_seq, qvalue_seq, not_done_seq = [], [], []
        for t in count():
            if args.mode == 'rl':
                action, qvalue = robot.act(state)
                # logger.info('Qvalue {}'.format(qvalue))
            elif args.mode == 'random':
                action, qvalue = random.randrange(num_actions), 0
            elif args.mode == 'uncertainty':
                # TODO uncertainty sampling not in use right now
                prob = state[-1]
                if prob > 0.2 or prob < 0.8:
                    action, qvalue = 1, 0
                else:
                    action, qvalue = 0, 0
            else:
                raise NotImplementedError
            _, next_state, done = game.step(action)
            state_seq += [state]
            act_seq += [action]
            next_state_seq += [next_state]
            qvalue_seq += [qvalue]

            not_done_seq += [1 - int(done)]

            if action > 0 and (game.chosen % game.duration == 0
                               or game.chosen == game.budget):
                print("-------------{}/{}-------------".format(game.chosen, game.budget))
                # Train the classifier
                train_acc1, val_acc1 = game.train_model(lr=lr, is_holdout=is_holdout)
                # select threshold
                test_acc1 = game.test_model(writer=writer,
                                            writer_name=str(i_episode) + '/threshold',
                                            duration=game.update,
                                            is_holdout=is_holdout)
                if not is_holdout:
                    # Evaluate on fixed set
                    fix_acc1, fix_sure_acc1 = fixed_set_evaluation(category, 'RL', i_episode, game.update)
                    writer.add_scalars(str(i_episode) + '/accuracy', {
                        'train_acc1': train_acc1,
                        'val_acc1': val_acc1,
                        'test_acc1': test_acc1,
                        'fix_acc1': fix_acc1,
                        'fix_sure_acc1': fix_sure_acc1*100},
                                       game.update)
                else:
                    writer.add_scalars(str(i_episode) + '/accuracy', {
                        'train_acc1': train_acc1,
                        'val_acc1': val_acc1,
                        'test_acc1': test_acc1},
                                       game.update)

                writer.add_scalar(str(i_episode) + '/lr', lr, game.update)

                if args.mode == 'rl' and not is_holdout:
                    # Read reward from difference from LSUN
                    reward = calculate_reward(category, i_episode, game.update)
                    game.current_reward = reward
                    writer.add_scalar(str(i_episode) + '/reward', reward, game.update)
                    logger.info('current reward in update {} of episode {} is {}'.format(
                        game.update, game.episode, game.current_reward))
                    reward_seq += [reward] * len(act_seq)

                    memory.push(state_seq, act_seq, next_state_seq,
                                reward_seq, qvalue_seq, not_done_seq)
                state_seq, act_seq, reward_seq = [], [], []
                next_state_seq, qvalue_seq, not_done_seq = [], [], []

                # train agent
                if len(memory) >= 2*args.batch_size \
                        and (args.mode == 'rl') and not is_holdout:
                    logger.info('updating robot. memory buffer size: {}'.format(len(memory)))
                    for x in range(args.batch_size):
                        _, _, next_state_batch, reward_batch, qvalue_batch, _ = memory.sample(1)
                        robot.update(next_state_batch, reward_batch, qvalue_batch, _,
                                     work_root='/data3/floraxue/cs294/exp/{0}/classifier'.format(args.exp))

            state = next_state

            if done:
                episode_durations.append(t + 1)
                # propagate through the whole dataset and split
                test_all_data(category, i_episode, new_key_path, is_holdout)

                # Set new key path
                new_key_path = join(MACHINE_LABEL_DIR, '{}_trial_{}_unsure.p'.format(category, trial))
                if args.mode == 'rl':
                    robot.target_q_function.load_state_dict(robot.q_function.state_dict())
                break


def adjust_learning_rate(iter):
    """ Sets the learning rate to the initial LR decayed by
    10 every 30 epochs """
    # lr = args.lr * (0.1 ** (epoch // 30))
    if iter == 1:
        return 5e-3
    elif iter >= 4:
        return 1e-4
    else:
        return 1e-3


def main():
    args = parse_arguments()
    game = VFGGAME(args)
    run_pipeline(args, game)


if __name__ == '__main__':
    main()
