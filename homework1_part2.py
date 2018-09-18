import gym
import deeprl_hw1.lake_envs as lake_env
import deeprl_hw1.rl as rl
import time
from matplotlib import pyplot as plt
import numpy as np

actions = lake_env.action_names
print(lake_env.action_names)
actions[lake_env.LEFT]= 'L'
actions[lake_env.RIGHT] = 'R'
actions[lake_env.UP] = 'U'
actions[lake_env.DOWN] = 'D'

gamma = 0.9
env = gym.make('Deterministic-4x4-FrozenLake-v0')
print("Result for policy_sync_4x4:")
policy, value_func, num_policy_iteration, num_value_iteration = rl.policy_iteration_sync(env,gamma)
print(num_value_iteration,num_policy_iteration)
rl.print_policy(policy,actions)
# values = np.reshape(value_func, (4, 4))
# plt.imshow(values)
# plt.colorbar()
# plt.show()
#print(policy)
# print("Result for policy_async_ordered_4x4:")
# policy, value_func, num_policy_iteration, num_value_iteration = rl.policy_iteration_async_ordered(env,gamma)
# print(policy)
# print(num_value_iteration,num_policy_iteration)
# print("Result for policy_async_randperm_4x4:")
# policy, value_func, num_policy_iteration, num_value_iteration = rl.policy_iteration_async_randperm(env,gamma)
# print(policy)
# print(num_value_iteration,num_policy_iteration)
# print("Result for value_sync_4x4:")
# policy, iteration, value_func = rl.value_iteration_sync(env,gamma)
# rl.print_policy(policy,actions)
# values = np.reshape(value_func, (4, 4))
# plt.imshow(values)
# plt.colorbar()
# plt.show()
# print(policy)
# print(iteration)
# print("Result for value_async_ordered_4x4:")
# policy, iteration = rl.value_iteration_async_ordered(env,gamma)
# print(policy)
# print(iteration)
# print("Result for value_async_randperm_4x4:")
# policy, iteration = rl.value_iteration_async_randperm(env,gamma)
# print(policy)
# print(iteration)
#
#
# env = gym.make('Deterministic-8x8-FrozenLake-v0')
# print("Result for policy_sync_8x8:")
# policy, value_func, num_policy_iteration, num_value_iteration = rl.policy_iteration_sync(env,gamma)
# print(num_value_iteration,num_policy_iteration)
# rl.print_policy(policy,actions)
# values = np.reshape(value_func, (8, 8))
# plt.imshow(values)
# plt.colorbar()
# plt.show()
#print(policy)
# print(num_value_iteration,num_policy_iteration)
# print("Result for policy_async_ordered_8x8:")
# policy, value_func, num_policy_iteration, num_value_iteration = rl.policy_iteration_async_ordered(env,gamma)
# print(policy)
# print(num_value_iteration,num_policy_iteration)
# print("Result for policy_async_randperm_8x8:")
# total_improve = 0
# total_eval = 0
# for i in range(10):
#     policy, value_func, num_policy_iteration, num_value_iteration = rl.policy_iteration_async_randperm(env,gamma)
#     total_improve +=  num_policy_iteration
#     total_eval += num_value_iteration
# print(policy)
# print(total_improve/10,total_eval/10)
# print("Result for value_sync_8x8:")
# policy, iteration, value_func = rl.value_iteration_sync(env,gamma)
# values = np.reshape(value_func, (8, 8))
# plt.imshow(values)
# plt.colorbar()
# plt.show()
# print(policy)
# print(iteration)
# print("Result for value_async_ordered_8x8:")
# policy, iteration = rl.value_iteration_async_ordered(env,gamma)
# print(policy)
# print(iteration)
# print("Result for value_async_randperm_8x8:")
# total_iteration = 0
# for i in range(10):
#     policy, iteration = rl.value_iteration_async_randperm(env,gamma)
#     total_iteration +=  iteration
# print(policy)
# print(total_iteration/10)

# env = gym.make('Stochastic-4x4-FrozenLake-v0')
# policy, iteration, value_func = rl.value_iteration_sync(env, gamma)
# print(policy)
# print(iteration)
# values = np.reshape(value_func, (4, 4))
# plt.imshow(values)
# plt.colorbar()
# plt.show()
#
# env = gym.make('Stochastic-8x8-FrozenLake-v0')
# policy, iteration, value_func = rl.value_iteration_sync(env, gamma)
# print(policy)
# print(iteration)
# values = np.reshape(value_func, (8, 8))
# plt.imshow(values)
# plt.colorbar()
# plt.show()

# tol = 1e-10
# max_iteration = 1e3
# #
# env = gym.make('Stochastic-4x4-FrozenLake-v0')
# policy, iteration, value_func = rl.value_iteration_sync(env, gamma,max_iteration,tol)
# print(policy)
# print(iteration)
# rl.print_policy(policy,actions)
# values = np.reshape(value_func, (4, 4))
# plt.imshow(values)
# plt.colorbar()
# plt.show()
#
# env = gym.make('Stochastic-8x8-FrozenLake-v0')
# policy, iteration, value_func = rl.value_iteration_sync(env, gamma,max_iteration,tol)
# print(policy)
# print(iteration)
# rl.print_policy(policy,actions)
# values = np.reshape(value_func, (8, 8))
# plt.imshow(values)
# plt.colorbar()
# plt.show()

###Custom & Stochastic
# env = gym.make('Deterministic-4x4-FrozenLake-v0')
# policy, iteration = rl.value_iteration_async_custom(env, gamma)
# print(policy)
# print(iteration)
#
# env = gym.make('Deterministic-8x8-FrozenLake-v0')
# policy, iteration = rl.value_iteration_async_custom(env, gamma)
# print(policy)
# print(iteration)
# tol = 1e-10
# max = 1e3
# env = gym.make('Stochastic-4x4-FrozenLake-v0')
# policy, iteration = rl.value_iteration_async_custom(env, gamma,max,tol)
# print(policy)
# print(iteration)
#
# env = gym.make('Stochastic-8x8-FrozenLake-v0')
# policy, iteration = rl.value_iteration_async_custom(env, gamma,max,tol)
# print(policy)
# print(iteration)