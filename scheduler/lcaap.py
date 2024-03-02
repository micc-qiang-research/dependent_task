import networkx as nx
import numpy as np
import portion as P
from util import *
from .scheduler import Scheduler
import math
from cluster import Core
from .executor import Executor

class LCAAP(Scheduler):
    def __init__(self, data, config):
        super().__init__(data, config)

    def schedule(self):
        self.sched = self.lcaa()
        print(self.sched)
        return self.output_scheduler_strategy()
    
    def output_scheduler_strategy(self):
        replica = False
        place = [[] for i in range(self.cluster.get_total_core_number())]
        download_sequence = None
        core_index = [0 for i in range(self.K)] # 计算当前正在使用server的core index

        for func,server in self.sched.items():
            place[core_index[server]+self.cluster.get_start_core_number(server)].append(func)
            core_index[server] += 1
            core_index[server] %= self.cluster.get_core_number(server)

        return replica, place, download_sequence, Executor.TOPOLOGY

    # 放置算法
    # 返回 {container_object_id: server_id, ...}
    def lcaa(self):
        deployment = {}

        max_comm = np.max(self.server_comm)

        # server部署的layer
        server_layer = [set() for i in range(self.K)]
        
        # server部署的func
        server_func = [[] for i in range(self.K)]

        # 为每个func选择一个server
        for func_id in nx.topological_sort(self.G):
            func = self.funcs[func_id]
            if func_id == self.source or func_id == self.sink:
                deployment[func_id] = self.generate_pos
                continue

            k_a = -1
            # score越小，优先级越高
            
            # 获取最大的传输边
            edges = self.G.in_edges(func_id, data=True)
            
            max_comm_cost = 0
            if len(edges) != 0:
                max_edge = max(edges, key=lambda t: t[2]['weight'])[2]['weight']
                max_comm_cost = max_edge * max_comm

            # print(self.get_func_total_layer_size(func_id))
            # exit(0)
            score_a = 2 * self.get_func_total_layer_size(func_id) * max([self.servers[k].download_latency for k in range(self.K)]) + max_comm_cost

            # print(score_a)
            # continue

            for k in range(self.K):
                if not self.can_deploy(server_layer[k], func_id, self.servers[k].storage):
                    continue

                L_inc = self.get_increment(server_layer[k], func_id)

                C_used = self.get_total_layer_size(server_layer[k])

                fetch_latency = self.servers[k].download_latency

                comm_cost = 0

                for j in self.G.predecessors(func_id):
                    if j in deployment and deployment[j] != k:
                        comm_cost = max(comm_cost, \
                        self.server_comm[deployment[j]][k]*self.G.edges[j,func_id]['weight'])

                score = self.score(L_inc, C_used, fetch_latency, comm_cost)
                if score < score_a:
                    k_a = k
                    score_a = score

            if k_a == -1:
                assert False, "No server can hold this image!"
            # 选定server[k_a]
            server_layer[k_a] |= set(func.layer)
            deployment[func_id] = k_a

        return deployment

    '''
    @Params:
        L_inc: 需要增加的layer大小
        C_used: 已经占用的空间大小
        fetch_latency: 镜像块传输时间
        comm_cost: 通信开销
    '''
    def score(self, L_inc, C_used, fetch_latency, comm_cost):
        alpha = 0.5
        beta = 0.5
        return (1-beta)*((1-alpha)*L_inc + alpha*C_used) * fetch_latency + beta*comm_cost


    # 获取某个函数layers的总大小
    def get_func_total_layer_size(self, func_id):
        return sum([self.layers[i].size for i in self.funcs[func_id].layer])
    
    # layers是目前含有的layer
    # func_id是即将部署的镜像（容器）
    def get_increment(self, layers, func_id):
        return sum([self.layers[i].size for i in self.funcs[func_id].layer if i not in layers])
    
    # 判断是否可以部署（不超过存储限制）
    def can_deploy(self, layers, func_id, limit):
        layers = layers | set(self.funcs[func_id].layer)
        return sum([self.layers[i].size for i in layers]) <= limit
    
    def get_total_layer_size(self, layers):
        return sum([self.layers[i].size for i in layers])
    