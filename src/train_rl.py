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
from train_new import MACHINE_LABEL_DIR, CLASSIFIER_ROOT
from train_new import test_fixed_set, test_all
import random
from tensorboardX import SummaryWriter

writer = SummaryWriter('runs/')
exp = ''

def parse_arguments():
    parser = argparse.ArgumentParser(description="training N-step Q learning")
    parser.add_argument('--category', type=str, default='cat',
                        help='image category')
    parser.add_argument('--budget', type=int, default=100,
                        help='maximum number of examples for human annotation')
    parser.add_argument('--eps-start', type=float, default=0.9,
                        help='starting epsilon')
    parser.add_argument('--eps-end', type=float, default=0.05,
                        help='ending epsilon')
    parser.add_argument('--decay-steps', type=int, default=100000,
                        help='decay steps')
    parser.add_argument('--gamma', type=float, default=0.999,
                        help='discount factor')
    parser.add_argument('--duration', '-k', type=int, default=20,
                        help='get reward every k steps')
    parser.add_argument('--batch-size', type=int, default=1,
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
    parser.add_argument('--feat-dir', type=str,
                        default='/data3/floraxue/cs294/active-rl-data/data/feats/train/cat')
    parser.add_argument('--gt-path', type=str,
                        default='/data3/floraxue/cs294/active-rl-data/ground_truth/cat_gt_cached.p')
    parser.add_argument('--val_rate', type=float, default=0.2)
    parser.add_argument('--test_rate', type=float, default=0.2)
    parser.add_argument('--mode', type=str, default='rl')
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--pretrained', type=str, default='')

    args = parser.parse_args()
    # global work_dir
    # work_dir = args.work_dirs
    args.key_path = '/data3/floraxue/cs294/active-rl-data/cat_trial_0_unsure.p'
    global exp
    exp = args.exp
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
    model_file_dir = join(CLASSIFIER_ROOT, 'latest_{}'.format(mode), 'snapshots')
    test_prefix = '{}_{}_episode_{:04d}_update_{:03d}'.format(mode, category, i_episode, update)
    return test_fixed_set(test_keys_path, method, category, test_prefix, model_file_dir)


def test_all_data(category, i_episode, last_trial_key_path):
    """
    test to split the dataset
    :return:
    """
    trial = i_episode
    mode = 'RL'
    model_file_dir = join(CLASSIFIER_ROOT, 'latest_{}'.format(mode), 'snapshots')
    # last_trial_key_path = join(MACHINE_LABEL_DIR,
    #                            '{}_trial_{}_unsure.p'.format(category, trial - 1))

    test_all(last_trial_key_path, trial, 'resnet', 'cat', model_file_dir)


def train_nsq(args, game, q_func):
    # build model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = args.input_dim
    num_actions = args.num_actions

    memory = ReplayMemory(args.buffer_size)
    Q = q_func(input_dim, num_actions)
    if args.pretrained != '':
        Q.load_state_dict(torch.load(args.pretrained))
    Q = Q.to(device)
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
    new_key_path = args.key_path

    last_acc = 0
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
            elif args.mode == 'lsun':
                action, qvalue = random.randrange(num_actions), 0
            elif args.mode == 'unsure':
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
                train_acc1, val_acc1 = game.train_model(train_mode='latest_RL',
                                                        work_root='/data3/floraxue/cs294/exp/{0}/classifier'.format(args.exp),
                                                        lr=lr)
                # select threshold
                test_acc1 = game.test_model(test_mode='latest_RL',
                                            work_root='/data3/floraxue/cs294/exp/{0}/classifier'.format(args.exp),
                                            writer=writer,
                                            name=str(i_episode) + '/threshold',
                                            duration=game.update)
                # Evaluate on fixed set
                fix_acc1, fix_sure_acc1 = fixed_set_evaluation(category, 'RL', i_episode, game.update)
                writer.add_scalars(str(i_episode) + '/accuracy', {
                    'train_acc1': train_acc1,
                    'val_acc1': val_acc1,
                    'test_acc1': test_acc1,
                    'fix_acc1': fix_acc1,
                    'fix_sure_acc1': fix_sure_acc1*100},
                                   game.update)

                writer.add_scalar(str(i_episode) + '/lr', lr, game.update)

                # train_lsun_model(game, 'latest_LSUN', '/data3/floraxue/cs294/active-rl-data/classifier')
                # test_lsun_model('latest_LSUN', '/data3/floraxue/cs294/active-rl-data/classifier')

                # Keep track of the place where last duration left off
                # game.last = game.index

                # fixed_set_evaluation(category, 'LSUN', i_episode, game.update)

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
                print(len(memory), args.batch_size, args.mode)
                # import pdb; pdb.set_trace()
                if len(memory) >= args.batch_size and (args.mode == 'rl'):
                    logger.info('updating robot. memory buffer size: {}'.format(len(memory)))
                    _, _, next_state_batch, reward_batch, \
                    qvalue_batch, _ = memory.sample(args.batch_size)
                    robot.update(next_state_seq, reward_batch, qvalue_batch, _,
                                 work_root='/data3/floraxue/cs294/exp/{0}/classifier'.format(args.exp))

            state = next_state

            if done:
                episode_durations.append(t + 1)
                # propagate through the whole dataset and split
                test_all_data(category, i_episode, new_key_path)

                # Set new key path
                new_key_path = join(MACHINE_LABEL_DIR, '{}_trial_{}_unsure.p'.format(category, trial))
                robot.target_q_function.load_state_dict(
                    robot.q_function.state_dict())
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
    q_func = NewPolicyNet
    train_nsq(args, game, q_func)


if __name__ == '__main__':
    main()
