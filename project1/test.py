#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mdp_dp import *
import gym
import sys
import numpy as np

from gym.envs.registration import register

"""
    This file includes unit test for mdp_dp.py
    You could test the correctness of your code by 
    typing 'nosetests -v mdp_dp_test.py' in the terminal
"""
env = gym.make("FrozenLake-v0")
env = env.unwrapped

register(
    id='Deterministic-4x4-FrozenLake-v0',
    entry_point='gym.envs.toy_text.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False})
env2 = gym.make("Deterministic-4x4-FrozenLake-v0")

#---------------------------------------------------------------


def test_python_version():
    '''------Dynamic Programming for MDP (100 points in total)------'''
    assert sys.version_info[0] == 3  # require python 2

#---------------------------------------------------------------


def test_policy_evaluation():
    '''policy_evaluation (20 points)'''
    random_policy1 = np.ones([env.nS, env.nA]) / env.nA
    #V1 = policy_evaluation(env.P, env.nS, env.nA, random_policy1)
    gamma = 1
    value_function = np.zeros(env.nS)
    next_value_function = value_function

    tol = 1e-3
    print(tol)
    while True:
        delta = 0
        for state in range(env.nS):  # loop through every state
            v = 0
            for action, action_probility in enumerate(random_policy1[state]):
                for probility, nextstate, reward, terminal in env.P[state][action]:
                    v += probility * action_probility *\
                        (reward + gamma * value_function[nextstate])
                    # update delta
            delta = max(delta, np.abs(
                        value_function[state] - v))
            value_function[state] = v
        if delta <= tol:
            print(value_function)
            break

        # if delta < theta:
        #     print(value_function)
        #     break

    test_v1 = np.array([0.004, 0.004, 0.01, 0.004, 0.007, 0., 0.026, 0., 0.019,
                        0.058, 0.107, 0., 0., 0.13, 0.391, 0.])
    print(test_v1)
    np.random.seed(595)
    random_policy2 = np.random.rand(env.nS, env.nA)
    random_policy2 = random_policy2/random_policy2.sum(axis=1)[:, None]
    #V2 = policy_evaluation(env.P, env.nS, env.nA, random_policy2)
    test_v2 = np.array([0.007, 0.007, 0.017, 0.007, 0.01, 0., 0.043, 0., 0.029,
                        0.093, 0.174, 0., 0., 0.215, 0.504, 0.])
    print(random_policy2)
    # assert np.allclose(test_v1, V1, atol=1e-3)
    # assert np.allclose(test_v2, V2, atol=1e-3)



#---------------------------------------------------------------
def test_policy_improvement():
    '''policy_improvement (20 points)'''
    np.random.seed(595)
    V1 = np.random.rand(env.nS)
    new_policy1 = policy_improvement(env.P, env.nS, env.nA, V1)
    print(new_policy1)
    print("######")
    test_policy1 = np.array([[1., 0., 0., 0.],
       [0., 0., 0., 1.],
       [0., 0., 0., 1.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.],
       [1., 0., 0., 0.],
       [0., 0., 1., 0.],
       [1., 0., 0., 0.],
       [0., 0., 0., 1.],
       [0., 0., 0., 1.],
       [0., 1., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [0., 0., 0., 1.],
       [0., 0., 1., 0.],
       [1., 0., 0., 0.]])
    print(test_policy1)
    V2 = np.zeros(env.nS)
    new_policy2 = policy_improvement(env.P, env.nS, env.nA, V2)
    #print(new_policy2)
    test_policy2 = np.array([[1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [1., 0., 0., 0.]])
    #print(test_policy2)
    # print(test_policy2)
    # print(new_policy2)
    assert np.allclose(test_policy1,new_policy1)
    assert np.allclose(test_policy2,new_policy2)
if __name__ == "__main__":
    test_policy_improvement()