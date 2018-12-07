import torch
import torch.optim as optim

from itertools import count
import argparse
from os.path import join
import pickle

from agent import NSQ
from policy import PolicyNet
from .buffer import ReplayMemory
from .game import VFGGAME
from .explorer import Explorer
from util import logger
from train_new import MACHINE_LABEL_DIR, CLASSIFIER_ROOT
import subprocess
from lsun import train_lsun_model, test_lsun_model


def parse_arguments():
    parser = argparse.ArgumentParser(description="training N-step Q learning")
    parser.add_argument('--category', type=str, default='cat',
                        help='image category')
    parser.add_argument('--budget', type=int, default=5000,
                        help='maximum number of examples for human annotation')
    parser.add_argument('--eps-start', type=float, default=0.9,
                        help='starting epsilon')
    parser.add_argument('--eps-end', type=float, default=0.05,
                        help='ending epsilon')
    parser.add_argument('--decay-steps', type=int, default=100000,
                        help='decay steps')
    parser.add_argument('--gamma', type=float, default=0.999,
                        help='discount factor')
    parser.add_argument('--duration', '-k', type=int, default=100,
                        help='get reward every k steps')
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
    parser.add_argument('--episodes', type=int, default=5)

    # flags for the game
    parser.add_argument('--eval-dir', type=str, default='',
                        help='path to the training list folder')
    parser.add_argument('--train-prefix', type=str, default='train',
                        help='prefix of the training files')
    parser.add_argument('--key-path', type=str,
                        default='/data/active-rl-data/machine_labels/cat_trial_0_unsure.p',
                        help='key path for the unknown data set')
    parser.add_argument('--feat-dir', type=str,
                        default='/data/active-rl-data/data/feats/train/cat')
    parser.add_argument('--gt-path', type=str,
                        default='/data/active-rl-data/ground_truth/cat_gt_cached.p')
    parser.add_argument('--val_rate', type=float, default=0.2)
    parser.add_argument('--test_rate', type=float, default=0.2)

    args = parser.parse_args()
    # global work_dir
    # work_dir = args.work_dirs
    return args


def fixed_set_evaluation(category, mode, i_episode, update):
    """
    Get test accuracy on a fixed set to calculate reward
    :param category:
    :param mode:
    :param i_episode:
    :param update:
    :return:
    """
    test_keys_path = '/data/active-rl-data/pool/{}_fixed_keys.p'.format(category)
    method = 'resnet'
    model_file_dir = join(CLASSIFIER_ROOT, 'latest_{}'.format(mode), 'snapshots')
    test_prefix = '{}_{}_episode_{:04d}_update_{:03d}'.format(mode, category, i_episode, update)
    cmd = 'python3 -m vfg.label.train_new test_fixed -e {0} ' \
          '-m {1} --category {2} --test-prefix {3} --model-file-dir {4}'.format(
            test_keys_path, method, category, test_prefix, model_file_dir)

    subprocess.check_call(cmd, shell=True, stderr=subprocess.STDOUT)


def calculate_reward(category, i_episode, update):
    """
    Calculate reward from LSUN and RL
    :return:
    """
    prefix = '{}_episode_{:04d}_update_{:03d}'.format(category, i_episode, update)
    lsun_file = join(CLASSIFIER_ROOT, 'fixed_set_acc_LSUN_{}.p'.format(prefix))
    rl_file = join(CLASSIFIER_ROOT, 'fixed_set_acc_RL_{}.p'.format(prefix))
    lsun_acc = pickle.load(open(lsun_file, 'rb'))
    rl_acc = pickle.load(open(rl_file, 'rb'))
    acc_gain = rl_acc - lsun_acc
    return acc_gain


def test_all_data(category, i_episode):
    """
    test to split the dataset
    :return:
    """
    trial = i_episode
    mode = 'RL'
    model_file_dir = join(CLASSIFIER_ROOT, 'latest_{}'.format(mode), 'snapshots')
    last_trial_key_path = join(MACHINE_LABEL_DIR,
                               '{}_trial_{}_unsure.p'.format(category, trial - 1))

    cmd = 'python3 -m vfg.label.train_new test_all -e {0} --trial {1}' \
          '-m {2} --category {3} --model-file-dir {4}'.format(
            last_trial_key_path, trial, 'resnet', 'cat', model_file_dir)

    output = subprocess.check_output(cmd, shell=True,
                                     stderr=subprocess.STDOUT)


def train_nsq(args, game, q_func):
    # build model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = args.input_dim
    num_actions = args.num_actions

    memory = ReplayMemory(args.buffer_size)
    Q = q_func(input_dim, num_actions).to(device)
    target_Q = q_func(input_dim, num_actions).to(device)
    # copy parameters of Q net to the target network
    target_Q.load_state_dict(Q.state_dict())

    optimizer = optim.RMSprop(Q.parameters())

    expr = Explorer(args.eps_start, args.eps_end, decay_steps=args.decay_steps)

    robot = NSQ(Q, target_Q, optimizer, expr,
                gamma=args.gamma, num_actions=num_actions)

    episode_durations = []

    # Pipeline params
    category = args.category
    # Set initial unsure key path
    new_key_path = join(MACHINE_LABEL_DIR, '{}_trial_{}_unsure.p'.format(category, 0))

    for i_episode in range(1, args.episodes + 1):
        game.reset(new_key_path)

        if len(game.train_data) < game.budget:
            logger.info("Unsure set is smaller than budget size. End this experiment.")
            break

        # pipeline param
        trial = i_episode

        # this is to reset the hidden units, not repackage the variables
        robot.q_function.reset_hidden()
        robot.target_q_function.reset_hidden()

        # sample the initial feature from the environment
        # since our policy network takes the hidden state and the current
        # feature as input. The hidden state is passed implicitly
        state = game.sample()
        state_seq, act_seq, reward_seq = [], [], []
        next_state_seq, qvalue_seq, done_seq = [], [], []
        for t in count():
            action, qvalue = robot.act(state)
            reward, next_state, done = game.step(action)
            state_seq += [state]
            act_seq += [action]
            next_state_seq += [next_state]
            qvalue_seq += [qvalue]
            done_seq += [done]

            if action > 0 and (game.chosen % game.duration == 0
                               or game.chosen == game.budget):
                # Train the classifier
                game.train_model()
                # select threshold
                game.test_model()
                # Evaluate on fixed set
                fixed_set_evaluation(category, 'RL', i_episode, game.update)

                train_lsun_model(game)
                test_lsun_model()

                # Keep track of the place where last duration left off
                game.last = game.index

                fixed_set_evaluation(category, 'LSUN', i_episode, game.update)

                # Read reward from difference from LSUN
                reward = calculate_reward(category, i_episode, game.update)
                game.current_reward = reward
                logger.info('current reward in update {} of episode {} is {}'.format(
                            game.update, game.episode, game.current_rewared))
                reward_seq += [reward]*len(act_seq)
                memory.push(state_seq, act_seq, next_state_seq,
                            reward_seq, qvalue_seq, done_seq)
                state_seq, act_seq, reward_seq = [], [], []
                next_state_seq, qvalue_seq, done_seq = [], [], []

                # train agent
                if len(memory) >= 5 * args.batch_size:
                    _, _, next_state_batch, reward_batch, \
                        qvalue_batch, not_done_batch = memory.sample(args.batch_size)
                    robot.update(next_state_seq, reward_batch, qvalue_batch, not_done_batch)

            state = next_state

            if done:
                episode_durations.append(t+1)
                # propagate through the whole dataset and split
                test_all_data(category, i_episode)

                # Set new key path
                new_key_path = join(MACHINE_LABEL_DIR, '{}_trial_{}_unsure.p'.format(category, trial))
                robot.target_q_function.load_state_dict(
                    robot.q_function.state_dict())
                break


def main():
    args = parse_arguments()
    game = VFGGAME(args)
    q_func = PolicyNet
    train_nsq(args, game, q_func)


if __name__ == '__main__':
    main()

