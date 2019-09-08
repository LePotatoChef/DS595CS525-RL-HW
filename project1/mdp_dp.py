# MDP Value Iteration and Policy Iteration
# Reference: https://web.stanford.edu/class/cs234/assignment1/index.html
import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """

    value_function = np.zeros(nS)
    ############################
    # YOUR IMPLEMENTATION HERE #
    #tol = 1e-8
    #prev_value_function = value_function.copy()
    # while(delta >= tol):  # terminate situation
    while True:
        delta = 0
        for state in range(nS):  # loop through every state
            v = 0
            # get probability distribution over actions
            for action, action_probility in enumerate(policy[state]):
                for probility, nextstate, reward, terminal in P[state][action]:
                    # apply bellman expectatoin eqn
                    v += action_probility * probility * \
                        (reward + gamma * value_function[nextstate])
            delta = max(delta, np.abs(  # update delta with the maximum change
                        v - value_function[state]))
            value_function[state] = v
        if delta < tol:
            break
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    Hints:
    ------
    You should construct a stochastic policy that puts equal probability on maximizing action
    """

    new_policy = np.ones([nS, nA]) / nA

    ############################
    # YOUR IMPLEMENTATION HERE #
    def one_step_lookahead(s, value_fn):
        A = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in P[s][a]:
                A[a] += prob * (reward + gamma * value_fn[next_state])
        return A
    policy = np.ones([nS, nA]) / nA
    action_values = np.zeros(nA)
    while True:
        policy_stable = True
        # loop over state space
        for s in range(nS):
            # perform one step lookahead
            actions_values = one_step_lookahead(s, value_from_policy)
            # maximize over possible actions
            best_action = np.argmax(actions_values)
            # best action on current policy
            chosen_action = np.argmax(policy[s])
            # if Bellman optimality equation not satisifed
            if(best_action != chosen_action):
                policy_stable = False

            # the new policy after acting greedily w.r.t value function
            policy[s] = np.eye(nA)[best_action]
        # if Bellman optimality eqn is satisfied
        if(policy_stable):
            return policy
    ############################
    # return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()
    ############################
    # YOUR IMPLEMENTATION HERE #
    V = np.zeros(nS)
    while True:
        V = policy_evaluation(P, nS, nA, policy)
        new_policy = policy_improvement(P, nS, nA, V)
        if np.all(new_policy == policy):
            break
        policy = new_policy
    ############################
    return new_policy, V


def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """

    V_new = V.copy()

    ############################
    # YOUR IMPLEMENTATION HERE #

    # Terminate state
    # max(value_function(s) - prev_value_function(s)) < tol

    ############################
    return policy_new, V_new


def render_single(env, policy, render=False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game.
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob = env.reset()  # initialize the episode
        done = False
        while not done:
            if render:
                env.render()  # render the game
            ############################
            # YOUR IMPLEMENTATION HERE #

    return total_rewards
