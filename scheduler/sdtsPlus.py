import networkx as nx
import numpy as np
import portion as P
from util import *
from .scheduler import Scheduler
import math
from cluster import Core
from .executor import Executor


'''
SDTS的加强版
1. 计算EFT时考虑镜像块共享
2. 没有副本
3. cloud部署任务也需要准备环境

'''
class SDTSPlus(Scheduler):
    def __init__(self, data, config):
        super().__init__(data, config)
        
        self.func_edge_download = self.func_download_time
        
        self.is_scheduler = set() # 目前已经调度的节点

    # 计算下载优先级，环境越大优先级越高
    def priority(self, func_edge_download, server_comm, func_process):
        G = self.G
        vertices = list(nx.topological_sort(G))
        vertices.reverse() # 计算优先级的次序
        N = self.data.N
        priority_dict = { i: 0 for i in range(N)}
        assert N == len(vertices), "N != len(vertices)"

        d_mean = np.sum(server_comm) / self.data.K**2 # edge server之间的平均传输数据延迟

        for v in vertices:
            if v == N - 1 or v == 0:
                continue
            t_mean = func_process[v].mean() # func v的平均处理时间
            t_d_mean = func_edge_download[v].mean() # func v的平均下载时间
            for s in G.successors(v):
                priority_dict[v] = max(priority_dict[v], self.get_weight( v, s) * d_mean + priority_dict[s])
            priority_dict[v] = priority_dict[v] + t_mean + t_d_mean
        self.logger.debug(priority_dict)
        return priority_dict

    def edge_server_selection(self, func, func_edge_download, func_prepare, func_process):
        early_start_time = P.inf
        early_core_id = None
        early_server_id = -1
        for idx, server in enumerate(self.cluster.get_server()):
            # t_e = self.cluster.get_download_complete(idx) + func_edge_download[func][idx] + func_prepare[func] # 环境准备好时间
            
            # 镜像块共享感知，下载时间为 增量*下载延迟
            t_e = self.cluster.get_download_complete(idx) + self.get_func_deploy_increment(idx, func) * self.servers[idx].download_latency

            t_i = self.get_input_ready(func, "edge", idx) # 数据依赖准备好时间
            t = max(t_e, t_i)
            # 得到在此服务器上的最早开始时间
            core, start_time = server.ESTfind(t, func_prepare[func], func_process[func][idx])
            if start_time < early_start_time:
                early_start_time = start_time
                early_core_id = core.idx
                early_server_id = idx

        ### 选择 early_idx对应的server 作为目标server       
        # 更新server的下载完成时间
        t_download_start = self.cluster.get_download_complete(early_server_id)
        t_download_finish = self.cluster.get_download_complete(early_server_id) + self.get_func_deploy_increment(early_server_id, func) * self.servers[early_server_id].download_latency
        self.cluster.set_download_complete(early_server_id, t_download_finish)

        # 将任务放置
        self.cluster.place(early_server_id, early_core_id, early_start_time - func_prepare[func], early_start_time + func_process[func][early_server_id])
        
        # 记录调度策略，策略记录的开始时间是开始执行时间，不包括准备时间
        self.strategy[func].deploy("edge", \
            server_id=early_server_id, \
            core_id=early_core_id, \
            t_download_start=t_download_start, \
            t_download_end=t_download_finish, \
            t_execute_start=early_start_time, \
            t_execute_end=early_start_time + func_process[func][early_server_id])

            
    def _all_successor_scheduler(self, node):
        return set(self.G.successors(node)).issubset(self.is_scheduler)

    def output_scheduler_strategy(self):
        total_core_number = self.cluster.get_total_core_number()
        replica = False # 该策略允许复制
        place = [[] for i in range(total_core_number)]
        download_sequence = None

        # 获取每个核的调度信息
        sched_info = [[] for i in range(total_core_number)]
        cloud_deploy = []
        for j in range(len(self.strategy)):
            strategy = self.strategy[j]
            if strategy.is_deploy("edge"):
                info = strategy.get_deploy_info("edge")
                server_id = info["server_id"]
                core_id = info["core_id"]
                pid = self.cluster.get_total_core_id(server_id, core_id)
                sched_info[pid].append({"func": j, "start": info["t_execute_start"]})
        
                    
        for i in range(total_core_number):
            sched_info[i] = sorted(sched_info[i], key=lambda x: x["start"])
            for sched in sched_info[i]:
                place[i].append(sched["func"])

        self.logger.debug(f"replica? {replica}")
        self.logger.debug(f"place: {place}")
        self.logger.debug(f"download_sequence: {download_sequence}")
        
        return replica, place, download_sequence, Executor.CUSTOM, self.sorted_nodes
            
    def schedule(self):
        G = self.G
        func_process = self.func_process
        func_edge_download = self.func_edge_download
        func_prepare = self.func_prepare
        server_comm = self.server_comm

        priority_dict =  self.priority(func_edge_download, server_comm, func_process)

        # 记录调度顺序
        self.sorted_nodes = []

        L = PQueue()
        L.put((-priority_dict[0],0))
        N = self.data.N
        
        while not L.empty():
            _, v = L.get()
            if v in self.is_scheduler: continue
            self.sorted_nodes.append(v)
            if v == 0 or v == N - 1:
                if v == 0:
                    self.gs(v).deploy("edge", server_id=self.generate_pos, core_id=0,t_execute_start=0, t_execute_end=0)
                else:
                    t_i = self.get_input_ready(v, "edge", self.generate_pos)
                    self.gs(v).deploy("edge", server_id=self.generate_pos, core_id=0, t_execute_start=t_i, t_execute_end=t_i)
            else:
                # 根据EST调度
                self.edge_server_selection(v, func_edge_download, func_prepare, func_process)

            self.is_scheduler.add(v)

            # update
            for s in G.successors(v):
                if set(G.predecessors(s)).issubset(self.is_scheduler) and s not in self.is_scheduler:
                    L.put((-priority_dict[s],s))

        return self.output_scheduler_strategy()

        # draw_dag(self.G_)
        # self.show_result("sdts")
