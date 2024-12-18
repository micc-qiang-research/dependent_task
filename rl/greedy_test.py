import gym
import random
import numpy as np
import torch
from .env_config import *

# env_name = 'CartPole-v0'

# np.random.seed(0)
# random.seed(0)
# env.seed(0)
# torch.manual_seed(0)

class GreedyAgent:
    def __init__(self, N, L):
        self.N = N
        self.L = L

    def parse_state(self, state):
        N = self.N
        L = self.L
        machine_state = state[:self.N * (3 * self.L + 3)]
        task_state = state[self.N * (3 * self.L + 3):]
        
        download_finish_time = []
        for i in range(N):
            download_finish_time.append(task_state[i*(5+L)+1]+task_state[i*(5+L)+2])
        return download_finish_time

    def take_action(self, state):
        download_finish_time = self.parse_state(state)
        return download_finish_time.index(min(download_finish_time))

env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# state: N*（4L+8） 
# N*(3L+3) 机器的信息
# N*(L+5) 任务的信息
N = action_dim-1
L = (state_dim // N - 8) // 4
print(f"{N} machines, {L} layers")
agent = GreedyAgent(N, L)

# 渲染环境
state = env.reset(seed=42)
done = False
total_reward = 0
while not done:
    # env.render()
    action = agent.take_action(state)
    if not env.valid_action(action):
        action = action_dim-1
    state, reward, done, _, = env.step(action)
    total_reward += reward
    # print(state, reward)

print(total_reward)

env.close()