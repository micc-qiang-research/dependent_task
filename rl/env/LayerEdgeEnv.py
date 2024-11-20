import gym
from gym import spaces
import numpy as np
import math

'''
Node: 有多少台机器（N）, 每台机器的信息：（带宽b_n, 存储d_n, cpu f_n）。包括cloud
Layer: 有多少个Layer(L)，每个layer的大小
Container: 共有多少种Container，也即请求的种类数
           每个container信息：（cpu f_c， layer info）

Traces: [trace1, trace2, ..]
其中每条trace:
每条trace长度固定, 假设为Len
然后每个step的信息为：(timestamp, container_id)
其中timestamp之间的间隔通过指数分布生成，假设两个请求之间间隔符合指数分布
container_id采用均匀分布生成

'''

class Data:
    # 机器的信息
    lo_bandwidth = 1
    hi_bandwidth = 10
    lo_storage = 500
    hi_storage = 2000
    lo_cpu = 1
    hi_cpu = 10

    # 请求信息
    lo_request_cpu = 1
    hi_request_cpu = 10
    request_interval = 5 # 请求到达的间隔

    # layer信息
    lo_layer_size = 5
    hi_layer_size = 10
    lo_func_layer_number = 5
    hi_func_layer_number = 10


    # 获取请求的到达时间
    def getRequestArrivals(self, interval, Len):
        intervals = np.random.exponential(interval, Len)
        arrivals = np.zeros(intervals.shape)
        for i in range(1, Len):
            arrivals[i] = arrivals[i-1]+intervals[i-1]
        return arrivals
    
    # 获取请求的container类型
    def getContainerTypes(self, Len):
        container_types = np.random.randint(0, self.C, Len)
        return container_types
    
    # 获取请求的信息
    def getTrace(self, Len):
        arrivals = self.getRequestArrivals(self.request_interval, Len)
        container_types = self.getContainerTypes(Len)
        return np.column_stack((arrivals, container_types))
        # print(np.column_stack((arrivals, container_types)))

    # 获取机器信息
    def getMachines(self, N):
        machines = []
        for i in range(N):
            machine = dict()
            machine['cpu'] = np.random.uniform(self.lo_cpu, self.hi_cpu)
            machine['storage'] = np.random.uniform(self.lo_storage, self.hi_storage)
            machine['bandwidth'] = np.random.uniform(self.lo_bandwidth, self.hi_bandwidth)
            machines.append(machine)
        # print(machines)
        return machines
    
    def getLayers(self, L):
        layers = np.zeros(L)
        for i in range(L):
            layers[i] = np.random.uniform(self.lo_layer_size, self.hi_layer_size)
        # print(layers)
        return layers
    
    def getContainerInfo(self, L, C):
        containers = []
        for i in range(C):
            container = dict()
            container['cpu'] = np.random.uniform(self.lo_request_cpu, self.hi_request_cpu)
            layer_number = np.random.randint(self.lo_func_layer_number, self.hi_func_layer_number+1)
            container['layer'] = np.random.choice(np.arange(0, L), layer_number, replace=False)
            container['layer'].sort()
            containers.append(container)
        # print(containers)
        return containers

    def getCloud(self):
        cloud = dict()
        cloud['cpu'] = np.random.uniform(self.lo_cpu, self.hi_cpu)
        cloud['storage'] = Math.INF
        cloud['bandwidth'] = np.random.uniform(self.lo_bandwidth, self.hi_bandwidth)
        return cloud

    def __init__(self, N, L, C, Len):
        self.N = N
        self.L = L
        self.C = C
        self.Len = Len
        self.machines = self.getMachines(N)
        self.layers = self.getLayers(L)
        self.containers = self.getContainerInfo(L, C)
        self.trace = self.getAnotherTrace()
        # return machines, layers, containers, trace
    
    def getAnotherTrace(self):
        return self.getTrace(self.Len)

np.random.seed(0)
data = Data(5, 50, 20, 100)
print(data.machines)
print(data.layers)
print(data.containers)
print(data.trace)
print("another: ")
print(data.getAnotherTrace())


class LayerEdgeEnv(gym.Env):
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