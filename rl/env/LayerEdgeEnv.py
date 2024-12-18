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
        cloud['cpu'] = self.hi_cpu
        cloud['storage'] = math.inf # cloud没有带宽限制
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
        self.cloud = self.getCloud()
        # return machines, layers, containers, trace
    
    def getAnotherTrace(self):
        return self.getTrace(self.Len)

np.random.seed(0)
data = Data(5, 50, 20, 100)
print(data.machines)
print(data.cloud)
print(data.layers)
print(data.containers)
print(data.trace)
print("another: ")
print(data.getAnotherTrace())

print("=================================")

class Task:
    def __init__(self, container: dict, layer_size: list):
        self.container = container
        self.cpu = container['cpu']
        self.layer = set(container['layer'])
        self.has_layer = []
        for i in range(len(layer_size)):
            if(i in self.layer):
                self.has_layer.append(1)
            else:
                self.has_layer.append(0)

def ceil2(value):
    return math.ceil(value*100)/100

import portion as P
# 按至少0.01的粒度occupy，否则会有数值问题
class Core:
    def __init__(self, idx):
        self.idx = idx # 机器的第几个core
        self.interval = P.closedopen(0, P.inf)

    # 占据核[start, end)的资源
    def occupy(self, start, end):
        i = P.closedopen(start, end)
        # 假设已经被别人占领了，则无法占领
        if not self.interval.contains(i):
            print(self.interval)
            print(start, end)
            assert False, "occupy error"

        # 占领核[start, end)
        self.interval = self.interval - P.closedopen(start, end)
    
    def release(self, start, end):
        i = P.closedopen(start, end)
        if not (self.interval & i).empty:
            assert False, "release error" # 释放的是已经占据的
        self.interval = self.interval | i

    def is_occupy(self, start, end) -> bool:
        i = P.closedopen(start, end)
        return not self.interval.contains(i)
    
    def find_est(self, start, end) -> bool:
        for i in self.interval:
            if i.lower <= start and i.upper >= end:
                return i.lower
        assert False, "never be there"


    def __repr__(self):
        return self.interval.__str__()

    def __str__(self):
        return self.interval.__str__()

    def __iter__(self):
        return self.interval.__iter__()

class Machine:
    def __init__(self, cpu: float, storage: float, bandwidth: float, layer_size: list, core_number:int, idx: int):
        self.cpu = cpu
        self.storage = storage
        self.bandwidth = bandwidth
        self.L = len(layer_size)
        self.layer_size = layer_size
        self.core_number = core_number
        self.idx = idx
        self.reset()

    def reset(self):
        self.layers = {} # 记录对应layers的下载完成时间
        self.download_finish_time = 0
        # self.tasks = []
        # self.task_finish_time = 0
        self.total_download_size = 0
        self.has_layer = [0] * self.L
        self.cores = [Core(i) for i in range(self.core_number)]

    def getRemainingDownloadTime(self, timestamp: float):
        res = []
        for i in range(self.L):
            # 如果没有下载过或者已经下载完成
            if self.has_layer[i] == 0 or self.layers[i] <= timestamp:
                res.append(timestamp)
            else:
                res.append(self.layers[i])
        return res
    
    # 判断是否还能容纳此任务
    def isAccommodate(self, task: Task):
        if self.total_download_size + self.getAddLayersSize(task) > self.storage:
            return False
        # TODO. container number的限制如何做？
        return True
    
    def findEstByCore(self, start, end):
        res = math.inf
        core_id = -1
        for idx, core in enumerate(self.cores):
            est = core.find_est(start, end)
            if res > est:
                res = est
                core_id = idx
        return core_id, res
    
    def place(self, core_id, start, end):
        # print(f"edge[{self.idx}-{core_id}] occupy: {start}-{end}")
        self.cores[core_id].occupy(start, end)

                
    def addTask(self, task: Task, timestamp: float):
        # self.tasks.append(task)
        add_layers = self.getAddLayers(task)
        # 计算Layer下载完成时间
        ready_time = timestamp
        for layer in add_layers:
            self.layers[layer] = self.download_finish_time + self.layer_size[layer]/self.bandwidth
            # 记录信息
            self.download_finish_time = self.layers[layer]
            self.has_layer[layer] = 1
            self.total_download_size += self.layer_size[layer]
        if len(add_layers) > 0:
            ready_time = max(ready_time, self.download_finish_time)

        ready_time = ceil2(ready_time)
        
        # 计算Task完成时间
        execute_time = ceil2(task.cpu / self.cpu)
        core_id, est = self.findEstByCore(ready_time, ready_time+execute_time)
        self.place(core_id, est, est+execute_time)
        # print(f"{timestamp:.2f}: executing task at [{est:.2f}-{est+execute_time:.2f}) in edge {self.idx}")
        # self.task_finish_time = max(self.download_finish_time, self.task_finish_time) + execute_time
        return est + execute_time

    def getAddLayers(self, task: Task):
        # 计算Layer下载完成时间
        layers = set(task.layer)
        add_layers = self.layers.keys() - layers
        return add_layers
    
    def getAddLayersSize(self, task: Task):
        return sum([self.layer_size[layer] for layer in self.getAddLayers(task)])

class Cloud(Machine):
    def __init__(self, cpu: float, storage: float, bandwidth: float, layer_size: list):
        super().__init__(cpu, storage, bandwidth, layer_size, 4, -1)

    def addTask(self, task: Task, timestamp: float):
        add_layers = self.getAddLayers(task)
        # 计算Layer下载完成时间
        ready_time = timestamp
        for layer in add_layers:
            self.layers[layer] = self.download_finish_time + self.layer_size[layer]/self.bandwidth
            # 记录信息
            self.download_finish_time = self.layers[layer]
            self.has_layer[layer] = 1
            self.total_download_size += self.layer_size[layer]
        if len(add_layers) > 0:
            ready_time = max(ready_time, self.download_finish_time)
        ready_time = ceil2(ready_time)

        # 计算Task完成时间
        execute_time = ceil2(task.cpu / self.cpu)
        est = ready_time
        # print(f"{timestamp:.2f}: executing task at [{est}-{est+execute_time}) in cloud")
        # self.task_finish_time = max(self.download_finish_time, self.task_finish_time) + execute_time
        return est + execute_time
    
    def isAccommodate(self, task: Task):
        return True

class LayerEdgeEnv(gym.Env):
    def __init__(self, render_mode="human"):
        self.data = Data(5, 50, 20, 100)
        N,L = data.N, data.L
        obs_dim = N * (3*L+3) + N * (L+5)
        act_dim = N+1

        self.observation_space = spaces.Box(
            low=0, high=math.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(act_dim)
        self.state = None

        self.machines = []
        for idx, machine in enumerate(data.machines):
            self.machines.append(Machine(machine['cpu'], machine['storage'], machine['bandwidth'], data.layers, 1, idx))
        # self.machines.append(Machine(data.cloud['cpu'], data.cloud['storage'], data.cloud['bandwidth'], data.layers))
        self.layers = data.layers # layer_size的信息
        self.cloud = Cloud(data.cloud['cpu'], data.cloud['storage'], data.cloud['bandwidth'], data.layers)

    def __getState(self):
        # 获取当前被调度的任务
        task = self.__getTask()
        if task is None:
            return [0] * self.observation_space.shape[0]

        # for machine
        state = []
        for machine in self.machines:
            state.extend(machine.has_layer)
            state.extend(machine.getRemainingDownloadTime(self.timestamp))
            state.extend(self.layers)
            state.append(machine.cpu)
            state.append(machine.bandwidth)
            state.append(machine.download_finish_time)
        
        # for current request
        for machine in self.machines:
            addLayerSize = machine.getAddLayersSize(task)
            state.append(addLayerSize) # 需要下载的大小
            state.append(addLayerSize / machine.bandwidth) # 需要下载的时间
            state.append(max(self.timestamp, machine.download_finish_time)-self.timestamp) # waiting time
            state.append(task.cpu/machine.cpu) # 计算时间
            state.extend(task.has_layer) # 包含的层
            state.append(task.cpu) # request cpu resource

        return state

    def reset(self):
        self.timestamp = 0
        self.trace_idx = 0
        self.data.getAnotherTrace() # 初始化新的trace
        for machine in self.machines:
            machine.reset()
        self.cloud.reset()
        return self.__getState()
    
    def __getTask(self) -> Task:
        if self.__idDone():
            return None
        task_info = self.data.trace[self.trace_idx]
        arrival_time, container_id = task_info[0], int(task_info[1])
        container = self.data.containers[container_id]
        self.timestamp = arrival_time
        return Task(container, self.layers)
    
    def __next(self):
        self.trace_idx += 1

    def step(self, action):
        reward = 0
        if action == self.data.N:
            # to cloud
            reward = -self.cloud.addTask(self.__getTask(), self.timestamp)
        else:
            # to edge
            reward = -self.machines[action].addTask(self.__getTask(), self.timestamp)
        # 到下一个task
        self.__next()
        return self.__getState(), reward, self.__idDone(),  {}
    
    # 判断动作是否合法，不合法需要重新sample
    def valid_action(self, action: int) -> bool:
        if(action == self.data.N):
            return True
        return self.machines[action].isAccommodate(self.__getTask())

    def render(self):
        print(self.state)

    def __idDone(self) -> bool:
        return self.trace_idx == self.data.trace.shape[0]
