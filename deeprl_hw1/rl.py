# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import copy as copy


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)
    if len(policy) == 16:
        policy_print = np.reshape(str_policy, (4, 4))
        print(policy_print)
    elif len(policy) == 64:
        policy_print = np.reshape(str_policy, (8, 8))
        print(policy_print)
    #print(str_policy)


def value_function_to_policy(env, gamma, value_function):
    """output action numbers for each state in value_function.

    parameters
    ----------
    env: gym.core.environment
      environment to compute policy for. must have ns(num of states), na(num of actions), and p(p[states][actions](prob,next_state,reward,is_terminal)`) as
      attributes.
    gamma: float
      discount factor. number in range [0, 1)
    value_function: np.ndarray
      value of each state.

    returns
    -------
    np.ndarray
      an array of integers. each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    # extract attributes from env.
    nS = env.nS
    nA = env.nA
    P = env.P
    # initialize policy matrix
    policy = np.zeros(nS, dtype=int)
    for state in range(nS):
        value = np.zeros(nA)
        for action in range(nA):
            state_value = 0
            for prob, nextstate, reward, is_terminal in P[state][action]:
                if is_terminal:
                    state_value += prob * reward
                else:
                    state_value += prob * (reward + gamma * value_function[nextstate])
            value[action] = state_value
        policy[state] = np.argmax(value)  # find the largest q_value
    return policy


def evaluate_policy_sync(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """performs policy evaluation.

    evaluates the value of a given policy.

    parameters
    ----------
    env: gym.core.environment
      the environment to compute value iteration for. must have ns,
      na, and p as attributes.
    gamma: float
      discount factor, must be in range [0, 1)
    policy: np.array
      the policy to evaluate. maps states to actions.
    max_iterations: int
      the maximum number of iterations to run before stopping.
    tol: float
      determines when value function has converged.

    returns
    -------
    np.ndarray, int
      the value for the given policy and the number of iterations till
      the value function converged.
    """
    # extract attributes from env.
    nS = env.nS
    P = env.P
    value_function_old = np.zeros(nS)  # initialize the value function
    iteration = 0
    delta = 1
    while delta > tol and iteration <= max_iterations:
        delta = 0
        iteration += 1
        value_function = np.zeros(nS)
        for state in range(nS):
            action = policy[state]
            v_old = value_function_old[state]
            state_value = 0
            for prob, nextstate, reward, is_terminal in P[state][action]:
                # if is_terminal:
                #     state_value += prob * reward
                # else:
                #     state_value += prob * (reward + gamma * value_function[nextstate])
              #state_value += prob * (reward + gamma * value_function[nextstate])
                state_value += prob * (reward + gamma * value_function_old[nextstate])
            value_function[state] = state_value #update value function
            delta = max(delta, abs(value_function[state] - v_old))
        # print('num of iteration' + str(iteration))
        # print(value_function_old)
        # print('\n')
        # print(value_function)
        value_function_old = copy.copy(value_function )#update old function

        #print(delta)
    return value_function, iteration


def evaluate_policy_async_ordered(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """performs policy evaluation.

    evaluates the value of a given policy by asynchronous dp.  updates states in
    their 1-n order.

    parameters
    ----------
    env: gym.core.environment
      the environment to compute value iteration for. must have ns,
      na, and p as attributes.
    gamma: float
      discount factor, must be in range [0, 1)
    policy: np.array
      the policy to evaluate. maps states to actions.
    max_iterations: int
      the maximum number of iterations to run before stopping.
    tol: float
      determines when value function has converged.

    returns
    -------
    np.ndarray, int
      the value for the given policy and the number of iterations till
      the value function converged.
    """
    nS = env.nS
    P = env.P
    value_function = np.zeros(nS)  # initialize the value function
    iteration = 0
    delta = 1
    while delta > tol and iteration <= max_iterations:
        delta = 0
        iteration += 1
        for state in range(nS):
            action = policy[state]
            state_value = 0
            for prob, nextstate, reward, is_terminal in P[state][action]:
                # if is_terminal:
                #     state_value += prob * reward
                # else:
                #     state_value += prob * (reward + gamma * value_function[nextstate])
              #state_value += prob * (reward + gamma * value_function[nextstate])
                state_value += prob * (reward + gamma * value_function[nextstate])
            delta = max(delta, abs(value_function[state] - state_value))
            value_function[state] = state_value  # update value function
        #print(delta)
    return value_function, iteration


def evaluate_policy_async_randperm(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """performs policy evaluation.

    evaluates the value of a policy.  updates states by randomly sampling index
    order permutations.

    parameters
    ----------
    env: gym.core.environment
      the environment to compute value iteration for. must have ns,
      na, and p as attributes.
    gamma: float
      discount factor, must be in range [0, 1)
    policy: np.array
      the policy to evaluate. maps states to actions.
    max_iterations: int
      the maximum number of iterations to run before stopping.
    tol: float
      determines when value function has converged.

    returns
    -------
    np.ndarray, int
      the value for the given policy and the number of iterations till
      the value function converged.
    """
    nS = env.nS
    P = env.P
    value_function = np.zeros(nS)  # initialize the value function
    rand_state = np.random.permutation(nS)
    iteration = 0
    delta = 1
    while delta > tol and iteration <= max_iterations:
        delta = 0
        iteration += 1
        for state in rand_state:
            action = policy[state]
            state_value = 0
            for prob, nextstate, reward, is_terminal in P[state][action]:
                # if is_terminal:
                #     state_value += prob * reward
                # else:
                #     state_value += prob * (reward + gamma * value_function[nextstate])
                # state_value += prob * (reward + gamma * value_function[nextstate])
                state_value += prob * (reward + gamma * value_function[nextstate])
            delta = max(delta, abs(value_function[state] - state_value))
            value_function[state] = state_value  # update value function
    return value_function, iteration


def improve_policy(env, gamma, value_func, policy):
    """performs policy improvement.

    given a policy and value function, improves the policy.

    parameters
    ----------
    env: gym.core.environment
      the environment to compute value iteration for. must have ns,
      na, and p as attributes.
    gamma: float
      discount factor, must be in range [0, 1)
    value_func: np.ndarray
      value function for the given policy.
    policy: dict or np.array
      the policy to improve. maps states to actions.

    returns
    -------
    bool, np.ndarray
      returns true if policy changed. also returns the new policy.
    """
    is_changed = False
    nS = env.nS
    better_policy = value_function_to_policy(env, gamma, value_func)
    for state in range(nS):
        old_action = policy[state]
        new_action = better_policy[state]
        if old_action != new_action:
            is_changed = True
    return is_changed, better_policy


def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """runs policy iteration.

    see page 85 of the sutton & barto second edition book.

    you should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.

    parameters
    ----------
    env: gym.core.environment
      the environment to compute value iteration for. must have ns,
      na, and p as attributes.
    gamma: float
      discount factor, must be in range [0, 1)
    max_iterations: int
      the maximum number of iterations to run before stopping.
    tol: float
      determines when value function has converged.

    returns
    -------
    (np.ndarray, np.ndarray, int, int)
       returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    policy_stable = True  # true for changed, false for unchanged
    num_policy_iteration = 0
    num_value_iteration = 0
    while policy_stable:
        value_func, num = evaluate_policy_sync(env, gamma, policy, max_iterations, tol)
        num_value_iteration += num
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        num_policy_iteration += 1
        if num_policy_iteration > max_iterations:
            break
    return policy, value_func, num_policy_iteration, num_value_iteration


def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    policy_stable = True  # true for changed, false for unchanged
    num_policy_iteration = 0
    num_value_iteration = 0
    while policy_stable:
        value_func, num = evaluate_policy_async_ordered(env, gamma, policy, max_iterations, tol)
        num_value_iteration += num
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        num_policy_iteration += 1
        if num_policy_iteration > max_iterations:
            break
    return policy, value_func, num_policy_iteration, num_value_iteration


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                    tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    policy_stable = True  # true for changed, false for unchanged
    num_policy_iteration = 0
    num_value_iteration = 0
    while policy_stable:
        value_func, num = evaluate_policy_async_randperm(env, gamma, policy, max_iterations, tol)
        num_value_iteration += num
        policy_stable, policy = improve_policy(env, gamma, value_func, policy)
        num_policy_iteration += 1
        if num_policy_iteration > max_iterations:
            break
    return policy, value_func, num_policy_iteration, num_value_iteration


def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    nS = env.nS
    nA = env.nA
    P = env.P
    policy = np.zeros(nS, dtype='int')
    value_function_old = np.zeros(nS)
    delta = 1
    iteration = 0
    while delta > tol and iteration <= max_iterations:
        delta = 0
        iteration += 1
        value_function = np.zeros(nS)
        for state in range(nS):
            value_max = -1
            v_old = value_function_old[state]
            for action in range(nA):
                state_value = 0
                for prob, nextstate, reward, is_terminal in P[state][action]:
                    state_value += prob * (reward + gamma * value_function_old[nextstate])
                if value_max < state_value:
                    value_max = state_value
                    policy[state] = action
            value_function[state] = value_max
            delta = max(delta, abs(value_function[state] - v_old))
        value_function_old = copy.copy(value_function) #update old value function
    #print(value_function)
    return policy, iteration, value_function


def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    nS = env.nS
    nA = env.nA
    P = env.P
    policy = np.zeros(nS, dtype='int')
    value_function = np.zeros(nS)
    delta = 1
    iteration = 0
    while delta > tol and iteration <= max_iterations:
        delta = 0
        iteration += 1
        for state in range(nS):
            value_max = -1
            v_old = value_function[state]
            for action in range(nA):
                state_value = 0
                for prob, nextstate, reward, is_terminal in P[state][action]:
                    state_value += prob * (reward + gamma * value_function[nextstate])
                if value_max < state_value:
                    value_max = state_value
                    policy[state] = action
            value_function[state] = value_max #update value function
            delta = max(delta, abs(value_function[state] - v_old))
    #print(value_function)
    return policy, iteration


def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    nS = env.nS
    nA = env.nA
    P = env.P
    policy = np.zeros(nS, dtype='int')
    value_function = np.zeros(nS)
    rand_state = np.random.permutation(nS)
    delta = 1
    iteration = 0
    while delta > tol and iteration <= max_iterations:
        delta = 0
        iteration += 1
        for state in rand_state:
            value_max = -1
            v_old = value_function[state]
            for action in range(nA):
                state_value = 0
                for prob, nextstate, reward, is_terminal in P[state][action]:
                    state_value += prob * (reward + gamma * value_function[nextstate])
                if value_max < state_value:
                    value_max = state_value
                    policy[state] = action
            value_function[state] = value_max #update value function
            delta = max(delta, abs(value_function[state] - v_old))
    #print(value_function)
    return policy, iteration


def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    nS = env.nS
    nA = env.nA
    P = env.P
    policy = np.zeros(nS, dtype='int')
    value_function = np.zeros(nS)
    #sort the state by manhattan distance
    if nS == 16:
        #goal is in (2,2)
        distance_manhattan = np.zeros(nS)
        for state in range(nS):
            distance_manhattan[state] = abs(int(state/4) - 1) + abs(state%4 - 1)
        manhattan_order = np.argsort(distance_manhattan)
    elif nS == 64:
        #goal is in (8,2)
        distance_manhattan = np.zeros(nS)
        for state in range(nS):
            distance_manhattan[state] = abs(int(state/8) - 7) + abs(state%8 - 1)
        manhattan_order = np.argsort(distance_manhattan)
    #print(manhattan_order)
    rand_state = np.random.permutation(nS)
    delta = 1
    iteration = 0
    while delta > tol and iteration <= max_iterations:
        delta = 0
        iteration += 1
        for state in manhattan_order:
        #for state in rand_state:
            value_max = -1
            v_old = value_function[state]
            for action in range(nA):
                state_value = 0
                for prob, nextstate, reward, is_terminal in P[state][action]:
                    state_value += prob * (reward + gamma * value_function[nextstate])
                if value_max < state_value:
                    value_max = state_value
                    policy[state] = action
            value_function[state] = value_max #update value function
            delta = max(delta, abs(value_function[state] - v_old))
    #print(value_function)
    return policy, iteration

