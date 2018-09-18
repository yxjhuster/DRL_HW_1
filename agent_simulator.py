import gym
import deeprl_hw1.rl as rl

def run_policy(env,policy,gamma):
    total_reward = 0
    step_num = 0
    state = env.reset()
    is_terminal = False
    while not is_terminal:
        nextstate, reward, is_terminal, debug_info = env.step(policy[state])
        total_reward += pow(gamma,step_num) * reward
        step_num +=1
        state = nextstate
    return total_reward

def stochastic_run(env, gamma, tol = 1e-3):
    total_reward = 0
    max_iterations = int(1e3)
    policy, iteration, value_func = rl.value_iteration_sync(env, gamma, max_iterations, tol)
    for i in range(100):
        #policy, iteration, value_func= rl.value_iteration_sync(env, gamma, max_iterations,tol)
        reward = run_policy(env, policy, gamma)
        total_reward += reward

    total_reward = total_reward / 100
    return total_reward



# env = gym.make('Deterministic-4x4-FrozenLake-v0')
# gamma = 0.9
# initial_state = env.reset()
# policy, iteration, value_func = rl.value_iteration_sync(env, gamma)
# total_reward = run_policy(env,policy,gamma)
# print("computed value: " + str(total_reward))
# print("simulated value: " + str(value_func[initial_state]))
# print('\n')
#
# env = gym.make('Deterministic-8x8-FrozenLake-v0')
# gamma = 0.9
# initial_state = env.reset()
# policy, iteration, value_func = rl.value_iteration_sync(env, gamma)
# total_reward = run_policy(env,policy,gamma)
# print("computed value: " + str(total_reward))
# print("simulated value: " + str(value_func[initial_state]))
# print('\n')
#
env = gym.make('Stochastic-4x4-FrozenLake-v0')
gamma = 0.9
initial_state = env.reset()
policy, iteration, value_func= rl.value_iteration_sync(env, gamma)
reward = stochastic_run(env, gamma)
tol = 1e-10
reward_tol = stochastic_run(env, gamma, tol)
print("computed value: " + str(reward))
print("computed value (tol = 1e-10): " + str(reward_tol))
print("simulated value: " + str(value_func[initial_state]))
print('\n')

# env = gym.make('Stochastic-8x8-FrozenLake-v0')
# gamma = 0.9
# initial_state = env.reset()
# policy, iteration, value_func= rl.value_iteration_sync(env, gamma)
# # reward = stochastic_run(env, gamma)
# tol = 1e-10
# reward_tol = stochastic_run(env, gamma, tol)
# # print("computed value: " + str(reward))
# print("computed value (tol = 1e-10): " + str(reward_tol))
# print("simulated value: " + str(value_func[initial_state]))
# print('\n')
