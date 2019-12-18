#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v mc_test.py' in the terminal.

    You don't have to follow the comments to write your code. They are provided
    as hints in case you need.
'''
#-------------------------------------------------------------------------


def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise

    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    action = 0
    [current_score, face_up, status] = observation
    if current_score >= 20:
        action = 0
    else:
        action = 1
    # action

    ############################
    return action


def mc_prediction(policy, env, n_episodes, gamma=1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop each episode
    for i_episode in range(1, 1+n_episodes):
        # initialize the episode
        state = env.reset()
    # generate empty episode list
        episode = []
    # loop until episode generation is done
        for i in range(100):
            action = policy(state)
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            else:
                state = next_state
        seperate_episode = set([tuple(eps[0]) for eps in episode])
        for s_eps in seperate_episode:
            for i, x in enumerate(episode):
                if x[0] == s_eps:
                    first_visit_pos = i
            G = sum([e[2]*gamma**idx for idx,
                     e in enumerate(episode[first_visit_pos:])])
            returns_sum[s_eps] += G
            returns_count[s_eps] += 1.0
            V[s_eps] = returns_sum[s_eps]*1.0/returns_count[s_eps]
    # Just taking the mean of all the returns got by taking this action when we were in this state.
    # unless state_t appears in states

    # update return_count

    # update return_sum

    # calculate average return for this state over all sampled episodes
    #V = {state: np.mean(reward_list) for state, reward_list in returns.items()}
    ############################
    return V


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
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    action = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    action[best_action] += (1 - epsilon)
    probs = action
    action = np.random.choice(np.arange(nA), p=probs)
    ############################
    return action


def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    """Monte Carlo control with exploring starts. 
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.    
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #
    epsilon = epsilon - (0.1/n_episodes)
    for i_episode in range(1, 1+n_episodes):
        state = env.reset()
        episode = []
        for i in range(100):
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            #probs = action
            #action = np.random.choice(np.arange(env.action_space.n), p=probs)
            next_state, reward, done, info = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            else:
                state = next_state

        seperate_episode = set([(tuple(x[0]), x[1]) for x in episode])

        for state, action in seperate_episode:
            for idx, e in enumerate(episode):
                if e[0] == state and e[1] == action:
                    first_visit_idx = idx
                    break
            pair = (state, action)
            G = sum([e[2]*(gamma**i)
                     for i, e in enumerate(episode[first_visit_idx:])])
            returns_sum[pair] += G
            returns_count[pair] += 1.0
            Q[state][action] = returns_sum[pair]*1.0/returns_count[pair]
    return Q
