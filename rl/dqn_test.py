import gym
from .agent.DQN import DQN
import random
import numpy as np
import torch
from .env_config import *

lr = 2e-3
# num_episodes = 500
num_episodes = 50
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
max_action_count = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

# env_name = 'CartPole-v0'

env = gym.make(env_name, render_modes="human")
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)


agent.load("model/dqn_model.pth")

print("模型已加载")

# # 渲染环境
# env = gym.make("CartPole-v1")
env = gym.make(env_name,render_modes="human")
state = env.reset()
done = False
total_reward = 0
while not done:
    # env.render()
    action_count = 1
    action = agent.take_action(state)
    while not env.valid_action(action) and action_count < max_action_count:
        action = agent.take_action(state)
        action_count += 1
    if not env.valid_action(action):
        action = action_dim-1

    state, reward, done, _, = env.step(action)
    total_reward += reward
    # print(state, reward)

print(total_reward)

env.close()