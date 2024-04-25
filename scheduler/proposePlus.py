import numpy as np
from typing import NamedTuple
from docplex.mp.model import Model
import networkx as nx
from .lpts import LPTS
from .executor import Executor
from heft import heft
import math

class ProposePlus(LPTS):
    
    def solve_layer_sequence_problem(self, n, func_ids, deploy_cores=None):
        if len(func_ids) == 0:
            return []
        # 获取所有需要下载的layer
        layer_ids = self.get_func_layers(func_ids)
        
        # 初始化常用量
        layer_size = len(layer_ids)
        func_size = len(func_ids)
        total_size = layer_size + func_size + 2 # 包括一个source、sink节点
        source = 0
        sink = total_size-1
        core_number = self.servers[n].core
        func_map = {func_id:idx for idx,func_id in enumerate(func_ids)}
        layer_map = {layer_id:idx for idx,layer_id in enumerate(layer_ids)}

        # 编号规则，0虚拟节点，指向其他任何节点
        # 1..layer_size layer节点编号
        # layer_size+1 .. total_size func节点编号
        def layer_id2idx(id):
            return layer_map[id]+1
        def layer_idx2id(idx):
            return layer_ids[idx-1]
        
        def func_id2idx(id):
            return func_map[id]+layer_size+1
        def func_idx2id(idx):
            return func_ids[idx-layer_size-1]

        # 1. ### 初始化G ######
        G = nx.DiGraph()

        # source指向所有其他节点
        for i in range(0, total_size):
            if i!=source:
                G.add_edge(source, i, weight=0)
            if i!=sink:
                G.add_edge(i, sink, weight=0)

        func_set = set(func_ids)
        for func_id in func_ids:
            # layer指向所有需要它的func节点
            for layer_id in self.funcs[func_id].layer:
                G.add_edge(layer_id2idx(layer_id), func_id2idx(func_id), weight=0)
            # func指向其后继结点
            for successor in self.G.successors(func_id):
                if successor in func_set:
                    G.add_edge(func_id2idx(func_id), func_id2idx(successor), weight=0)


        # 2. ### 计算process ####
        INF = math.inf
        process = np.zeros((total_size, core_number+1))

        # 获取layer的处理时间：在核上为INF，在download线程为下载时间
        for i in range(1, layer_size+1):
            for k in range(core_number):
                process[i][k] = INF
            process[i][core_number] = self.servers[n].download_latency * self.layers[layer_idx2id(i)].size

        # 获取func的处理时间，在核上为func_process时间，在download线程为下载时间
        for i in range(layer_size + 1, total_size-1):
            for k in range(core_number):
                if deploy_cores:
                    if(deploy_cores[func_idx2id(i)] == k):
                        process[i][k] = self.func_process[func_idx2id(i)][n]
                    else:
                        process[i][k] = INF
                else:
                    process[i][k] = self.func_process[func_idx2id(i)][n]
            process[i][core_number] = INF

        self.sorted_nodes ,self.sched, self.task_sched, _ = heft.schedule_dag(G, 
                            communication_matrix=np.ones((core_number+1,core_number+1)), 
                            computation_matrix=process,communication_startup=np.zeros(core_number+1))
        
        # self.logger.debug(self.sched, self.task_sched)
        # print(self.sched[core_number])
        # print(self.task_sched)

        layer_sequence = []
        for i in self.sched[core_number]:
            layer_sequence.append(layer_idx2id(i.task))

        self.logger.debug(layer_sequence)
        return layer_sequence

    def parse_deploy(self, deploy):
        deploy_ = [[] for i in range(self.K)]
        for i,d in enumerate(deploy):
            deploy_[d].append(i)
        
        layer_sequence = []
        for i,d in enumerate(deploy_):
            layer_sequence.append(self.solve_layer_sequence_problem(i, d))
        return layer_sequence

    def output_scheduler_strategy(self):
        replica = False
        place = [[] for i in range(self.cluster.get_total_core_number())]
        download_sequence = self.layer_sequence
        core_index = [0 for i in range(self.K)] # 计算当前正在使用server的core index

        for func,server in enumerate(self.deploy):
            place[core_index[server]+self.cluster.get_start_core_number(server)].append(func)
            core_index[server] += 1
            core_index[server] %= self.cluster.get_core_number(server)
        # place = [[0,1],[3],[1,2],[2]]
        return replica, place, download_sequence, Executor.TOPOLOGY

    def __init__(self, data, config):
        super().__init__(data, config)

    def schedule(self):
        # 生成self.deploy
        self.iter_solve_deploy_model()
        self.layer_sequence = self.parse_deploy(self.deploy)
        return self.output_scheduler_strategy()
        
