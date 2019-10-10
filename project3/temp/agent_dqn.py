#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from agent import Agent
from dqn_model import DQN

import math
from torch.autograd import Variable

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 30000
TARGET_UPDATE = 10
steps_done = 0


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example:
            paramters for neural network
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN, self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.env = env
        self.batch_size = BATCH_SIZE
        self.gamma = 0.999
        self.eps_start = EPS_START
        self.eps_decay = EPS_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE

        self.policy_net = DQN(self.env.action_space.n)
        self.target_net = DQN(self.env.action_space.n)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        if use_cuda:
            self.policy_net.cuda()
            self.target_net.cuda()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-5)
        self.memory = deque(maxlen=10000)

        if args.test_dqn:
            # you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        global steps_done
        self.policy_net.eval()
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            return self.policy_net(
                Variable(torch.from_numpy(observation), volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
        else:
            return LongTensor([[random.randrange(self.env.action_space.n)]])
        ###########################
        return action

    def push(self, s, a, r, s_, done):
        """ You can add additional arguments as you need.
        Push new data to buffer and remove the old one if the buffer is full.

        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.memory.append((s, a, r, s_, done))
        if len(self.memory) > self.maxlen:
            self.replay_memory_store.popleft()
        self.memory_counter += 1

        ###########################

    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        #print("memory", len(self.memory), self.BATCH_SIZE)
        minibatch = random.sample(self.memory, self.BATCH_SIZE)
        minibatch = np.array(minibatch).transpose(0, 3, 1, 2)
        minibatch = torch.tensor(minibatch / 255.0)
        ###########################
        return minibatch

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]),
                                         volatile=True).cuda()
        state_batch = Variable(torch.cat(batch.state)).cuda()
        action_batch = Variable(torch.cat(batch.action)).cuda()
        reward_batch = Variable(torch.cat(batch.reward)).cuda()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        self.policy_net.train()
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(
            torch.zeros(BATCH_SIZE).type(Tensor)).cuda()
        next_state_values[non_final_mask] = self.target_net(
            non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * GAMMA) + reward_batch
        # Undo volatility (which was used to prevent unnecessary gradients)
        expected_state_action_values = Variable(
            expected_state_action_values.data).cuda()

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        num_episodes = 1400000
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            observation = self.env.reset()
            observation = observation.transpose((2, 0, 1))
            observation = observation[np.newaxis, :]
            state = observation

            for t in count():
                # Select and perform an action
                action = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action[0, 0])
                next_state = next_state.transpose((2, 0, 1))
                next_state = next_state[np.newaxis, :]
                reward = Tensor([reward])

                # Store the transition in memory
                self.memory.push(torch.from_numpy(state), action,
                                 torch.from_numpy(next_state), reward)

                # Observe new state
                if not done:
                    state = next_state
                else:
                    state = None

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    print('resetting env. episode %d \'s reward total was %d.' % (
                        i_episode+1, t+1))
                    break
            # Update the target network
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if i_episode % 50 == 0:
                checkpoint_path = os.path.join('save_dir', 'model-best.pth')
                torch.save(self.policy_net.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
        ###########################
