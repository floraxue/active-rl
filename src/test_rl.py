import sys
import torch
import torch.optim as optim

from itertools import count
import argparse
from os.path import join

from agent import NSQ
from policy import PolicyNet
from game import VFGGAME
from explorer import Explorer
from util import logger
from train_new import MACHINE_LABEL_DIR_HOLDOUT, CLASSIFIER_ROOT_HOLDOUT
from lsun import train_lsun_model, test_lsun_model_holdout, train_lsun_model_holdout

import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser(description="training N-step Q learning")
    parser.add_argument('--category', type=str, default='cat',
                        help='image category')
    parser.add_argument('--budget', type=int, default=10000,
                        help='maximum number of examples for human annotation')
    parser.add_argument('--eps-start', type=float, default=0.9,
                        help='starting epsilon')
    parser.add_argument('--eps-end', type=float, default=0.05,
                        help='ending epsilon')
    parser.add_argument('--decay-steps', type=int, default=100000,
                        help='decay steps')
    parser.add_argument('--gamma', type=float, default=0.999,
                        help='discount factor')
    parser.add_argument('--duration', '-N', type=int, default=100,
                        help='get reward every N steps')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--target-update', '-T', type=int, default=1000,
                        help='update target network every T steps')
    parser.add_argument('--learning-start', type=int, default=50000)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--num-actions', type=int, default=2,
                        help='default action is `keep` or `drop`')
    parser.add_argument('--input_dim', type=int, default=2048,
                        help='feature size')
    parser.add_argument('--save-every', type=int, default=1,
                        help='save the checkpoint every K episode')
    parser.add_argument('--val_rate', type=float, default=0.2)
    parser.add_argument('--test_rate', type=float, default=0.2)
    # flags for the game
    parser.add_argument('--eval-dir', type=str, default='',
                        help='path to the training list folder')
    parser.add_argument('--train-prefix', type=str, default='train',
                        help='prefix of the training files')
    parser.add_argument('--key-path', type=str,
                        help='key path for the unknown data set')
    parser.add_argument('--work-dir', type=str, default='', help = 'work dir')
    parser.add_argument('--pretrained', type=str, default='', help='path to pretrained NSQ policy')

    args = parser.parse_args()
    global work_dir
    work_dir = args.work_dirs
    return args


def test_nsq(args, game, q_func):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = args.input_dim
    num_actions = args.num_actions

    Q = q_func(input_dim, num_actions).to(device)
    target_Q = q_func(input_dim, num_actions).to(device)

    optimizer = optim.RMSprop(Q.parameters())

    expr = Explorer(args.eps_start, args.eps_end, decay_steps=args.decay_steps)

    robot = NSQ(Q, target_Q, optimizer, expr,
                gamma=args.gamma, num_actions=num_actions)

    episode_durations = []

    # Pipeline params
    category = args.category
    # Set initial unsure key path
    new_key_path = join(MACHINE_LABEL_DIR_HOLDOUT, '{}_trial_{}_unsure.p'.format(category, 0))

    # Test on RL agent
    logger.info('Testing on RL agent')
    for i_episode in range(1, args.episodes + 1):
        game.reset(new_key_path)

        # pipeline param
        trial = i_episode

        robot.q_function.reset_hidden(args.batch_size)
        robot.target_q_function.reset_hidden(args.batch_size)

        # sample the initial feature from the environment
        # since our policy network takes the hidden state and the current
        # feature as input. The hidden state is passed implicitly
        state = game.sample()
        for t in count():
            action, qvalue = robot.act(state)
            reward, next_state, done = game.step(action)

            if action > 0 and (game.chosen % game.duration == 0
                               or game.chosen == game.budget):
                # Train the classifier
                game.train_model('latest_RL', CLASSIFIER_ROOT_HOLDOUT)
                # select threshold
                game.test_model('latest_RL', CLASSIFIER_ROOT_HOLDOUT)

            state = next_state

            if done:
                episode_durations.append(t + 1)
                # propagate through the whole dataset and split
                test_all_data_holdout(category, i_episode, "RL")
                new_key_path = join(MACHINE_LABEL_DIR_HOLDOUT, 'RL', '{}_trial_{}_unsure.p'.format(category, trial))
                break

    # Test on LSUN
    logger.info("Testing on LSUN")
    for i_episode in range(1, args.episodes + 1):
        trial = i_episode
        new_key_path = join(MACHINE_LABEL_DIR_HOLDOUT, 'latest_LSUN',
                            '{}_trial_{}_unsure.p'.format('cat', trial - 1))
        train_lsun_model_holdout(game, "latest_LSUN", CLASSIFIER_ROOT_HOLDOUT, new_key_path)
        test_lsun_model_holdout("latest_LSUN", CLASSIFIER_ROOT_HOLDOUT)

        test_all_data_holdout(category, i_episode, "LSUN")


def test_all_data_holdout(category, i_episode, mode):
    """
    test to split the dataset
    :return:
    """
    trial = i_episode
    model_file_dir = join(CLASSIFIER_ROOT_HOLDOUT, 'latest_{}'.format(mode), 'snapshots')
    last_trial_key_path = join(MACHINE_LABEL_DIR_HOLDOUT, mode,
                               '{}_trial_{}_unsure.p'.format(category, trial - 1))

    cmd = 'python3 train_new.py test_all -e {0} --trial {1}' \
          '-m {2} --category {3} --model-file-dir {4}'.format(
            last_trial_key_path, trial, 'resnet', 'cat', model_file_dir)

    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as exc:
        print("Status : FAIL", exc.returncode, exc.output)
        sys.exit(-1)

def main():
    args = parse_arguments()
    game = VFGGAME(args)
    q_func = PolicyNet
    test_nsq(args, game, q_func)


if __name__ == '__main__':
    main()
