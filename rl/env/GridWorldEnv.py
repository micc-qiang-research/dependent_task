import gym
from gym import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    def __init__(self, render_mode="human"):
        # (0,0) -> (9,9)
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(4)  # 上、下、左、右
        self.state = None

    def reset(self):
        self.state = np.array([0, 0], dtype=np.int32)
        return self.state

    def step(self, action):
        moves = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        self.state = np.clip(self.state + np.array(moves[action]), 0, 9)
        done = np.all(self.state == 9)
        reward = 100 if done else -1
        return self.state, int(reward), done,  {}

    def render(self):
        print(self.state)