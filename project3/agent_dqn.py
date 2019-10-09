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
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


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
        self.BATCH_SIZE = 32
        self.maxlen = 100000
        self.memory = deque(maxlen=100000)
        self.memory_counter = 0

        self.EPSILON = 0.9
        self.GAMMA = 0.9
        self.N_ACTIONS = env.action_space.n
        self.N_STATES = env.observation_space.shape[0]
        ##
        self.in_channels = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        # Net Initilization
        self.eval_net, self.target_net = DQN(self.N_STATES, self.N_ACTIONS).to(device), DQN(
            self.N_STATES, self.N_ACTIONS).to(device)  # Q net and Target Q net
        # Loss and opitmizer
        self.optimizer = torch.optim.Adam(
            self.eval_net.parameters(), lr=1.5e-4)
        self.loss_func = nn.SmoothL1Loss()
        # Training parameters
        self.training_step_counter = 0
        self.update_q_step = 5000
        self.learn_step = 5000

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
        # sample = random.random()
        # if sample < self.EPSILON:
        #     with torch.no_grad():
        #         return policy_net(state).max(1)[1].view(1, 1)
        # else:
        #     return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)
        x = torch.unsqueeze(torch.FloatTensor(observation).to(device), 0)
        x.to(device)
        x = x.permute(0, 3, 1, 2)  # *from NHWC to NCHW*
        x.to(device)
        if np.random.uniform() < self.EPSILON:  # greedy policy
            actions_value = self.eval_net.forward(x)
            # action_ = torch.max(actions_value, 1)[1].data.to("cpu")
            # print(torch.max(actions_value, 1)[1].data.to("cpu").numpy()[0])
            action = torch.max(actions_value, 1)[1].data.to("cpu").numpy()[0]
            #print("Action", action)
        else:
            action = np.random.randint(0, self.num_actions)
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
        if len(self.memory) < self.BATCH_SIZE:
            return
        b_s, b_a, b_r, b_s_, done = self.replay_buffer()
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.GAMMA * \
            q_next.max(1)[0].view(self.BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def learn(self):
        if self.training_step_counter % self.update_q_step == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.training_step_counter += 1

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        for i_episode in range(400):
            s = self.env.reset()
            while True:
                a = self.make_action(s)  # device problem
                s_, r, done, info = self.env.step(a)
                self.push(s, a, r, s_, done)
                if self.training_step_counter >= 5000:
                    if self.training_step_counter % self.learn_step == 0:
                        print("step", self.training_step_counter)
                        self.learn()
                if done:
                    break
                s = s_
        ###########################
