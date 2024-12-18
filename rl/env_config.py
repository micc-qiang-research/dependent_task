import gym
from .env.GridWorldEnv import GridWorldEnv
from .env.LayerEdgeEnv import LayerEdgeEnv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# gym.register(
#     id='GridWorld-v0',
#     entry_point='__main__:GridWorldEnv',
# )
# env_name = 'GridWorld-v0'

gym.register(
    id='LayerEdgeEnv-v0',
    entry_point='__main__:LayerEdgeEnv'
)
env_name = 'LayerEdgeEnv-v0'
