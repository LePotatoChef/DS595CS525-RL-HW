#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v td_test.py' in the terminal.

    You don't have to follow the comments to write your code. They are provided
    as hints in case you need.
'''
#-------------------------------------------------------------------------


def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    action = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    action[best_action] += (1 - epsilon)
    probs = action
    action = np.random.choice(np.arange(nA), p=probs)
    # print(action)
    ############################
    return action


def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """

    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    ############################
    # YOUR IMPLEMENTATION HERE #
    done = False
    for i in range(1, 1+n_episodes):
        epsilon = 0.99*epsilon
        state = env.reset()
        action = epsilon_greedy(Q, state, env.action_space.n)
        while True:
            new_state, reward, done, info = env.step(action)
            new_action = epsilon_greedy(
                Q, new_state, env.action_space.n, epsilon)
            Q[state][action] = Q[state][action]+alpha * \
                (reward+gamma*Q[new_state]
                    [new_action]-Q[state][action])
            state = new_state
            action = new_action
            if done:
                break
    ############################
    return Q


def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #
    for i in range(1, 1 + n_episodes):
        state = env.reset()
        done = False
        while True:
            action = epsilon_greedy(Q, state, env.action_space.n)
            new_state, reward, done, info = env.step(action)
            #new_action = epsilon_greedy(Q, new_state, env.action_space.n)
            new_action = np.argmax(Q[state])
            Q[state][action] = Q[state][action]+alpha * \
                (reward+gamma*Q[new_state][new_action]-Q[state][action])
            state = new_state
            if done:
                break
    # loop n_episodes

    # initialize the environment

    # loop for each step of episode

    # get an action from policy

    # return a new state, reward and done

    # TD update
    # td_target with best Q

    # td_error

    # new Q

    # update state

    ############################
    return Q
