### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

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

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """

    value_function = np.zeros(nS)

    ############################
    # YOUR IMPLEMENTATION HERE #
    MAX_ITERATION = 1000000
    for step in range(MAX_ITERATION):
        prev_value_function = value_function.copy()
        for state in range(nS):
            value_function[state] = 0
            for probability, nextstate, reward, terminal in P[state][policy[state]]:
                value_function[state] += probability * (reward + gamma * prev_value_function[nextstate])
        if np.max(np.abs(value_function - prev_value_function)) < tol:
            break
    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

    new_policy = np.zeros(nS, dtype='int')

    ############################
    # YOUR IMPLEMENTATION HERE #
    def argmax_all(array):
        max_ids = np.argwhere(array == np.amax(array))
        return max_ids.flatten()
    
    for state in range(nS):
        Q = np.zeros(nA)
        for action in range(nA):
            for probability, nextstate, reward, terminal in P[state][action]:
                Q[action] += probability * (reward + gamma * value_from_policy[nextstate])
        optim_action_set = argmax_all(Q)
        new_policy[state] = optim_action_set[np.random.randint(optim_action_set.size)]
    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    # initialize policy with random actions
    # policy = np.random.randint(nA, size=nS)
    # but make sure that you are not standing still
    for state in range(nS):
        valid_action_set = np.zeros(0)
        for action in range(nA):
            probability, nextstate, reward, terminal = P[state][action][0]
            # if you are not definitely at the same state after taking the action
            if nextstate != state or probability != 1 or terminal == True:
                valid_action_set = np.append(valid_action_set, action)
        policy[state] = valid_action_set[np.random.randint(valid_action_set.size)]
    
    MAX_ITERATION = 1000000
    value_function = policy_evaluation(P, nS, nA, policy)
    for step in range(MAX_ITERATION):
        prev_value_function = value_function.copy()
        # it's better that the first policy improvement is based on an estimated value function,
        # instead of a random value function
        policy = policy_improvement(P, nS, nA, value_function, policy)
        value_function = policy_evaluation(P, nS, nA, policy)
        if np.max(np.abs(value_function - prev_value_function)) < tol:
            break
    ############################
    return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    def argmax_all(array):
        max_ids = np.argwhere(array == np.amax(array))
        return max_ids.flatten()
    
    MAX_ITERATION = 1000000
    for step in range(MAX_ITERATION):
        prev_value_function = value_function.copy()
        for state in range(nS):
            Q = np.zeros(nS)
            # try all possible actions and find the current optimal action
            for action in range(nA):
                for probability, nextstate, reward, terminal in P[state][action]:
                    Q[action] += probability * (reward + gamma * prev_value_function[nextstate])
            optim_action_set = argmax_all(Q)
            policy[state] = optim_action_set[np.random.randint(optim_action_set.size)]
            value_function[state] = np.max(Q)
        if np.max(np.abs(value_function - prev_value_function)) < tol:
            break
    ############################
    return value_function, policy

def render_single(env, policy, max_steps=100):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
        Environment to play on. Must have nS, nA, and P as
        attributes.
    Policy: np.array of shape [env.nS]
        The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    env.render();
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

    # comment/uncomment these lines to switch between deterministic/stochastic environments
    env = gym.make("Deterministic-4x4-FrozenLake-v0")
    # env = gym.make("Stochastic-4x4-FrozenLake-v0")

    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)

    print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_vi, 100)


