import numpy as np
import math
from strategy import Strategy
from cluster import Cluster
import networkx as nx

class Executor:
    DUMB = 0
    TOPOLOGY = 1
    def get_gen_strategy(self, strategy):
        match(strategy):
            case Executor.DUMB:
                return self.dumb_gen_strategy
            case Executor.TOPOLOGY:
                return self.topology_gen_strategy
            case _:
                assert False, "gen_strategy error"


    def __init__(self, data):
        self.data = data
        self.read_info()

        self.cluster = Cluster([i.core for i in self.data.servers])
        self.strategy = [Strategy(i,self.data.N) for i in range(self.N)]
        self.func_download_time = self.get_func_download_time(self.func_startup, np.array([s.download_latency for s in self.servers])) # 函数在各个边缘下载时间

    # 从data中读数据
    def read_info(self):
        self.N = self.data.N
        self.K = self.data.K
        self.L = self.data.L
        self.source = 0
        self.sink = self.N - 1
        self.G = self.data.G
        self.func_process = np.array(self.data.func_process) # 函数执行时间
        self.server_comm = np.array(self.data.server_comm) # 服务器之间的通信时间
        self.generate_pos = self.data.generate_pos # dag生成位置
        self.funcs = self.data.funcs
        self.servers = self.data.servers
        self.layers = self.data.layers
        self.func_prepare = np.array([0 for i in self.funcs]) # 函数环境准备时间
        self.func_startup = np.array(self.data.func_startup) # 函数环境大小

    # 获取某个函数的调度决策
    def gs(self, func):
        return self.strategy[func]
    
    # 默认以最后一台server作为cloud
    def get_cloud_id(self):
        return self.data.K-1
    
    # 得到边的权重
    def get_weight(self, s, d):
        res = self.G.edges[s,d]["weight"]
        return res
    
    def input_ready(self, func1, func2, pos1, pos2):
        weight = self.get_weight(func1, func2)

        if not self.gs(func1).is_deploy(pos1):
            return math.inf
        server1_info = self.gs(func1).get_deploy_info(pos1)
        server2_info = self.gs(func2).get_deploy_info(pos2) 
        
        data_trans_time = self.server_comm[server1_info["server_id"]][server2_info["server_id"]] * weight
        
        return server1_info["t_execute_end"] + data_trans_time

    def get_input_ready(self, func2, pos2, server_id):
        self.gs(func2).deploy(pos2, server_id) # 模拟
        res = 0
        for i in self.G.predecessors(func2):
            v = 1e10
            for pos in self.gs(i).get_all_deploys_names():
                v = min(v, self.input_ready(i, func2, pos, pos2))
            res = max(v, res)
        self.gs(func2).clear(pos2)
        return res
    
    # 根据函数环境大小即边缘带宽，计算下载时间
    def get_func_download_time(self, func_startup, edge_bandwith):
        return (np.tile(func_startup.reshape(len(func_startup), 1), (1, self.data.K)).T * edge_bandwith.reshape(-1,1)).T
    
    def get_makespan(self):
        return self.gs(self.sink).get_deploy_info("edge")["t_execute_end"]
    
    '''
    将调度算法策略转换为本项目策略
    通用的算法策略用一个二维数组表示
    strategy[K] = [
          [0,1,5],
          [2,4,3],
           ...
    ]
    其中strategy[k][i]表示第i个core的第j个任务
    core的编号顺序为: 
        server0.core0  server0.core1 ...
        server1.core0  server1.core1 ...
        ...
        cloud
    返回:
        (func, procs)元组， procs是一个list,因为一个函数可部署到多个位置
    '''

    # 按核顺序来返回函数
    def dumb_gen_strategy(self, raw_strategy):
        assert len(raw_strategy) >= self.cluster.get_total_core_number(), "strategy length don't match core number"
        finished_func = set()
        all_func = set(self.G.nodes())
        pos = [0 for i in range(len(raw_strategy))]
        while True:
            # 所有节点都遍历过
            if len(all_func ^ finished_func) == 0: break
            for i,s in enumerate(raw_strategy):
                if pos[i] >= len(s):
                    continue
                for j in range(pos[i], len(s)):
                    if set(self.G.predecessors(s[j])).issubset(finished_func):
                        pos[i] = j + 1
                        finished_func.add(s[j])
                        yield s[j],[i]
    
    # 按拓扑排序返回函数
    def topology_gen_strategy(self, raw_strategy):
        def find_pos(func):
            res = []
            for i,s in enumerate(raw_strategy):
                if func in s:
                    res.append(i)
                if func == self.source or func == self.sink:
                    return [-1]
            return res
        func = list(nx.topological_sort(self.G))
        for f in func:
            yield f,find_pos(f)

    # server含有哪些layer
    def __server_layer(self, server_id):
        if not hasattr(self, "server_layer"):
            self.server_layer = [set() for i in range(self.data.K)]
        return self.server_layer[server_id]
    
    # 添加一个layer
    def __server_add_layer(self, server_id, layer_id):
        if not hasattr(self, "server_layer"):
            self.server_layer = [set() for i in range(self.data.K)]
        self.server_layer[server_id].add(layer_id)
        # assert sum([self.layers[l].size for l in self.server_layer[server_id]]) <= self.servers[server_id].storage, "out-of-store"

    # 得到某台server空闲空间
    def __server_free_storage_size(self, server_id, virtual_capacity=None):
        capacity = self.servers[server_id].storage
        if virtual_capacity:
            capacity = virtual_capacity
        return capacity - sum([self.layers[l].size for l in self.server_layer[server_id]])

    # 判断server能否部署func
    def is_func_can_deploy(self, server_id, func_id, virtual_capacity=None):
        need_layer = self.funcs[func_id].layer - self.__server_layer(server_id)
        if sum([self.layers[l].size for l in need_layer]) <= self.__server_free_storage_size(server_id,  virtual_capacity):
            return True
        return False
    
    def func_deploy(self, server_id, func_id):
        need_layer = set(self.funcs[func_id].layer) - self.__server_layer(server_id)
        for l in need_layer:
            self.__server_add_layer(server_id, l)

    # 部署函数然后检查是否超出限制
    def func_deploy_and_check(self, server_id, func_id, virtual_capacity=None):
        self.func_deploy(server_id, func_id)
        return self.__server_free_storage_size(server_id, virtual_capacity) <= 0
    
    def get_server_deploy_size(self, server_id):
        return sum([self.layers[l].size for l in self.__server_layer(server_id)])
    
    def get_func_total_size(self, func_id_arr):
        layer = set()
        for i in func_id_arr:
            layer = layer.union(self.funcs[i].layer)
        return sum([self.layers[l].size for l in layer])