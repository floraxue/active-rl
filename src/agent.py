import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import random
import shutil

class NSQ(object):
    """
    Here we are going to implement the synchronous q learning

    Asynchronous N-step Q-Learning.
        See http://arxiv.org/abs/1602.01783
        Args:
            - q_function (A3CModel): Model to train
            - optimizer (chainer.Optimizer): optimizer used to train the model
            - explorer (Explorer): Explorer to use in training
        """
    def __init__(self, q_function, target_q_function, optimizer,
                 explorer,  gamma=0.999, num_actions=2):
        """
        Initialize the arguments

        """
        self.q_function = q_function
        self.target_q_function = target_q_function

        # copy the params of target to the original q
        self.target_q_function.load_state_dict(q_function.state_dict())

        self.optimizer = optimizer
        self.explorer = explorer

        self.t = 0    # current time stamp
        self.gamma = gamma
        self.num_actions = num_actions

    def act(self, state):
        """
        :param state:
        :return: action in one-hot encoding
        """
        sample = random.random()
        eps_threshold = self.explorer.value(self.t)

        state = torch.from_numpy(state).unsqueeze(0).cuda()
        qvalue = self.q_function(state)

        if sample > eps_threshold:
            action_index = torch.argmax(qvalue).item()
        else:
            action_index = random.randrange(self.num_actions)

        self.t += 1
        return action_index, qvalue

    def update(self, next_state_batch, past_rewards, past_action_values,
               not_done_mask, work_root):
        """

        different from the asynchronous n-step q learning,
        here we take a batch of next_state_batches and do the batch update
        """

        # gather the last q values
        next_state_batch = torch.FloatTensor(next_state_batch).cuda()
        past_rewards = torch.FloatTensor(past_rewards).cuda()
        past_action_values = torch.cat(past_action_values).max(1)[0]

        with torch.no_grad():
            next_max_q = self.target_q_function(next_state_batch).max(1)[0]
            # R = not_done_mask * next_max_q
            R = next_max_q

        bsz = next_state_batch.size(0)
        loss = 0
        duration = past_rewards.size(0)
        for i in reversed(range(duration)):
            R *= self.gamma
            R += past_rewards
            loss += F.smooth_l1_loss(past_action_values, R.clone())

        loss /= duration
        average_loss = loss.mean()
        self.optimizer.zero_grad()
        average_loss.backward(retain_graph = True)

        # gradient clipping
        for param in self.q_function.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        #save state_dict
        torch.save(self.q_function.state_dict(), os.path.join(work_root, 'agent_state_dict.pth'))


