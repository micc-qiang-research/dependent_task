import gym
import random
import numpy as np
import torch
from .env_config import *
from .agent.DQN import DQN

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

    def take_action(self, state):
        pass

class GreedyAgent(Agent):
    def __init__(self, env):
        super(GreedyAgent, self).__init__(env)
        # state: N*（4L+8） 
        # N*(3L+3) 机器的信息
        # N*(L+5) 任务的信息
        self.N = self.action_dim-1
        self.L = (self.state_dim // self.N - 8) // 4
        print(f"{self.N} machines, {self.L} layers")
        

    def parse_state(self, state):
        N = self.N
        L = self.L
        machine_state = state[:N * (3 * L + 3)]
        task_state = state[N * (3 * L + 3):]
        
        download_finish_time = []
        for i in range(N):
            download_finish_time.append(task_state[i*(5+L)+1]+task_state[i*(5+L)+2])
        return download_finish_time

    def take_action(self, state):
        download_finish_time = self.parse_state(state)
        action = download_finish_time.index(min(download_finish_time))
        if not self.env.valid_action(action):
            action = self.action_dim-1
        return action

class DQNAgent(Agent):
    def __init__(self, env):
        super(DQNAgent, self).__init__(env)

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
        self.max_action_count = 10
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # random.seed(0)
        # np.random.seed(0)
        # env.seed(0)
        # torch.manual_seed(0)
        agent = DQN(self.state_dim, hidden_dim, self.action_dim, lr, gamma, epsilon,
                    target_update, device)

        agent.load("model/dqn_model.pth")

        self.agent = agent

        print("模型已加载")

    def take_action(self, state):
        action_count = 1
        action = self.agent.take_action(state)
        while not self.env.valid_action(action) and action_count < self.max_action_count:
            action = self.agent.take_action(state)
            action_count += 1
        if not self.env.valid_action(action):
            action = self.action_dim-1
        return action


# random.seed(0)
# np.random.seed(0) # 为了保证机器是一致的
# env.seed(0)
# torch.manual_seed(0)
env = gym.make(env_name, render_modes="human")

greedy = GreedyAgent(env)
dqn = DQNAgent(env)
agents = [greedy, dqn]
agent_names = ["greedy", "dqn"]

# # 渲染环境
# env = gym.make("CartPole-v1")

results = {}
results['greedy'] = []
results['dqn'] = []
for i in range(10):
    for idx, agent in enumerate(agents):
        state = env.reset(seed=i)
        done = False
        total_reward = 0
        while not done:
            # env.render()
            action = agent.take_action(state)
            state, reward, done, _, = env.step(action)
            total_reward += reward
            # print(state, reward)
        results[agent_names[idx]].append(total_reward)
        # print(total_reward)
env.close()

print(results)