import gym
from .env.GridWorldEnv import GridWorldEnv

gym.register(
    id='GridWorld-v0',
    entry_point='__main__:GridWorldEnv',
)
env_name = 'GridWorld-v0'
