import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import random


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

        state = state.unsqueeze(0).cuda()

        qvalue, hidden_unit = self.q_function(state)
        if sample > eps_threshold:
            action_index = torch.argmax(qvalue)
        else:
            action_index = random.randrange(self.num_actions)

        self.t += 1
        return action_index, qvalue

    def update(self, next_state_batch, past_rewards, past_action_values,
               not_done_mask):
        """

        different from the asynchronous n-step q learning,
        here we take a batch of next_state_batches and do the batch update
        """

        # gather the last q values
        with torch.no_grad:
            next_max_q = self.target_q_function(next_state_batch).max(1)[0]
            R = not_done_mask * next_max_q

        bsz = next_state_batch.size(0)
        loss = torch.zeros(bsz)
        duration = past_rewards.size(1)
        for i in reversed(range(duration)):
            R *= self.gamma
            R += past_rewards[:, i]
            current_q = past_action_values[:, i]
            loss += F.smooth_l1_loss(current_q, R)

        loss /= duration
        average_loss = loss.mean()

        self.optimizer.zero_grad()
        average_loss.backward()

        # gradient clipping
        for param in self.q_function.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


