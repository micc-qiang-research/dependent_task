import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import portion as P
from data import Data
from util import PQueue, draw_dag
from cluster import EdgeServer
from strategy import SchedStrategy

class SDTS:
    def __init__(self, data):
        self.data = data
        self.G_ = nx.DiGraph()
        self.G_end = "end"
        self.edge_server = [EdgeServer(i, 2) for i in range(self.data.K - 1)]
        self.G = self.data.G
        self.func_process = np.array(self.data.func_process) # 函数执行时间
        self.edge_bandwith = np.array(self.data.edge_bandwidth) # 边缘服务器带宽 1..(k-1)
        self.func_startup = np.array(self.data.func_startup) # 函数环境大小
        self.func_edge_download = self.get_func_edge_download(self.func_startup, self.edge_bandwith) # 函数在各个边缘下载时间
        self.func_prepare = np.array(self.data.func_prepare)
        self.server_comm = np.array(self.data.server_comm) # 服务器之间的通信时间
        self.ue_comm = np.array(self.data.ue_comm) # 用户<->edge
        self.uc_comm = self.data.uc_comm # 用户<->cloud

        self.t_server_download_complete = [0] * (self.data.K - 1) # edge server当前下载完成时间
        self.N = self.data.N
        self.K = self.data.K
        self.source = 0
        self.sink = self.N - 1
        self.pos_user = 0
        self.pos_edge = 1
        self.pos_cloud = 2
        self.is_scheduler = set() # 目前已经调度的节点
        
        self.strategy = [SchedStrategy(i, self.data.N) for i in range(self.data.N)] # 任务放置的服务器、核、开始时间、结束时间

    ########## deploy strategy 得到某个func的部署edge位置
    # function strategy
    def gs(self, func):
        return self.strategy[func]

    ###################################################

    # 根据函数环境大小即边缘带宽，计算下载时间
    def get_func_edge_download(self, func_startup, edge_bandwith):
        return (np.tile(func_startup.reshape(len(func_startup), 1), (1, self.data.K-1)).T / edge_bandwith.reshape(-1,1)).T
    
    # 得到边的权重
    def get_weight(self, s, d):
        res = self.G.edges[s,d]["weight"]["weight"]
        return res

    def priority(self, func_edge_download, server_comm, func_process):
        G = self.G
        vertices = list(nx.topological_sort(G))
        vertices.reverse() # 计算优先级的次序
        N = self.data.N
        priority_dict = { i: 0 for i in range(N)}
        assert N == len(vertices), "N != len(vertices)"

        d_mean = np.sum(server_comm[:-1,:-1]) / (self.data.K - 1)**2 # edge server之间的平均带宽

        for v in vertices:
            if v == N - 1 or v == 0:
                continue
            t_mean = func_process[v][:-1].mean() # func v的平均处理时间
            t_d_mean = func_edge_download[v].mean() # func v的平均下载时间
            for s in G.successors(v):
                priority_dict[v] = max(priority_dict[v], self.get_weight( v, s) * d_mean + priority_dict[s])
            priority_dict[v] = priority_dict[v] + t_mean + t_d_mean
        print(priority_dict)
        return priority_dict


    def input_ready(self, func1, func2, pos1, pos2):
        weight = self.get_weight(func1, func2)
        match pos1:
            case self.pos_user:
                assert func1 == self.source, "error"
                match pos2:
                    case self.pos_user: # user -> user
                        assert False, "error"
                    case self.pos_cloud: # user -> cloud
                        return self.uc_comm * weight
                    case self.pos_edge: # user -> edge
                        server = self.gs(func2).get_edge_id()
                        return self.ue_comm[server] * weight
            case self.pos_edge:
                match pos2:
                    case self.pos_user: # edge -> user
                        assert func2 == self.sink, "error"
                        server = self.gs(func1).get_edge_id()
                        return self.gs(func1).get_edge_end() + self.ue_comm[server] * weight
                    case self.pos_cloud: # edge -> cloud
                        server = self.gs(func1).get_edge_id()
                        return self.gs(func1).get_edge_end() + self.server_comm[server][self.K - 1] * weight
                    case self.pos_edge: # edge -> edge
                        s1 = self.gs(func1).get_edge_id()
                        s2 = self.gs(func2).get_edge_id()
                        return self.gs(func1).get_edge_end() + self.server_comm[s1][s2] * weight
            case self.pos_cloud:
                match pos2:
                    case self.pos_user: # cloud -> user
                        return self.gs(func1).get_cloud_end() + self.uc_comm * weight
                    case self.pos_cloud: # cloud -> cloud
                        return self.gs(func1).get_cloud_end()
                    case self.pos_edge: # cloud -> edge
                        server = self.gs(func2).get_edge_id()
                        return self.gs(func1).get_cloud_end() + self.server_comm[server][self.K-1]
            case _:
                assert False, "input ready error"

    def get_input_ready(self, func2, pos2):
        res = 0
        for i in self.G.predecessors(func2):
            if i == self.source:
                v = self.input_ready(i, func2, self.pos_user, pos2)
            else:
                v = min(
                    self.input_ready(i, func2, self.pos_edge, pos2),
                    self.input_ready(i, func2, self.pos_cloud, pos2),
                )
            res = max(v, res)
        return res

    def edge_server_selection(self, func, func_edge_download, func_prepare, func_process):
        early_start_time = P.inf
        early_core = None
        early_idx = -1
        for idx, server in enumerate(self.edge_server):
            self.gs(func).deploy_in_edge(idx) # 尝试deploy

            t_e = self.t_server_download_complete[idx] + func_edge_download[func][idx] + func_prepare[func] # 环境准备好时间
            
            # t_i = self.get_input_ready(idx, func) # 数据依赖准备好时间
            t_i = self.get_input_ready(func, self.pos_edge) # 数据依赖准备好时间
            t = max(t_e, t_i)
            # 得到在此服务器上的最早开始时间
            core, start_time = server.ESTfind(t, func_prepare[func], func_process[func][idx])
            if start_time < early_start_time:
                early_start_time = start_time
                early_core = core
                early_idx = idx

            self.gs(func).clear_edge_deploy() # 清除deploy

        ### 选择 early_idx对应的server 作为目标server       
        # 更新server的下载完成时间
        t_download_start = self.t_server_download_complete[early_idx]
        t_download_finish = self.t_server_download_complete[early_idx] + func_edge_download[func][early_idx]
        self.t_server_download_complete[early_idx] = t_download_finish

        # 将任务放置
        self.edge_server[early_idx].place(early_core, early_start_time - func_prepare[func], early_start_time + func_process[func][early_idx] )
        
        # 记录调度策略，策略记录的开始时间是开始执行时间，不包括准备时间
        self.strategy[func].deploy_in_edge(early_idx, early_core.idx, \
            t_download_start=t_download_start, \
            t_download_end=t_download_finish, \
            t_prepare_start=early_start_time - func_prepare[func], \
            t_prepare_end=early_start_time,\
            t_execute_start=early_start_time, \
            t_execute_end=early_start_time + func_process[func][early_idx])


    def _update_G_(self, G_, source, dest):
        if source == self.source:
            assert dest != self.sink, "error"
            G_.add_edge(source, dest)
            G_.add_edge(source, -dest)
        elif dest == self.sink:
            if self.input_ready(source, dest, self.pos_edge, self.pos_user) <= \
                self.input_ready(source, dest, self.pos_cloud, self.pos_user):
                G_.add_edge(source, dest)
            else:
                G_.add_edge(-source, dest)
        else:
            # dest
            if self.input_ready(source, dest, self.pos_edge, self.pos_edge) <= \
                self.input_ready(source, dest, self.pos_cloud, self.pos_edge):
                G_.add_edge(source, dest)
            else:
                G_.add_edge(-source, dest)  

            # -dest
            if self.input_ready(source, dest, self.pos_edge, self.pos_cloud) <= \
                self.input_ready(source, dest, self.pos_cloud, self.pos_cloud):
                G_.add_edge(source, -dest)
            else:
                G_.add_edge(-source, -dest)  
        # draw_dag(G_)
            

    def _all_successor_scheduler(self, node):
        return set(self.G.successors(node)).issubset(self.is_scheduler)

    # 新func添加后，更新G_
    def task_refinement(self, G_, G, dest):
        L_c = [] # 等待被clean的节点
        for source in G.predecessors(dest):
            self._update_G_(G_, source, dest)
            if self._all_successor_scheduler(source):
                G_.remove_edge(source, self.G_end)
                L_c.append(source)
                if source != self.source:
                    G_.remove_edge(-source, self.G_end)
                    L_c.append(-source)
        
        while L_c:
            node = L_c.pop()

            # already delete
            if node not in G_.nodes: continue

            if len(list(G_.successors(node))) == 0:
                for source in G_.predecessors(node):
                    # G_.remove_edge(source, node)
                    L_c.append(source)
                # print("--- remove node : ", node)
                G_.remove_node(node) # 一个node代表一个部署到云或者边缘的策略
                if node < 0:
                    self.gs(-node).clear_cloud_deploy()
                    # TODO cloud的resource如何管理？
                else:
                    self.gs(node).clear_edge_deploy()
                    edge_parms = self.gs(node).edge_param
                    self.edge_server[edge_parms["id"]].release(edge_parms["core"], edge_parms["t_prepare_start"] , edge_parms["t_execute_end"])


    def sdts(self):
        G = self.G
        func_process = self.func_process
        edge_bandwith = self.edge_bandwith
        func_startup = self.func_startup
        func_edge_download = self.func_edge_download
        func_prepare = self.func_prepare
        server_comm = self.server_comm

        priority_dict =  self.priority(func_edge_download, server_comm, func_process)

        L = PQueue()
        L.put((-priority_dict[0],0))
        N = self.data.N
        h = [-1] * N
        self.G_.add_node(self.G_end)
        
        while not L.empty():
            _, v = L.get()
            if v == 0 or v == N - 1:
                if v == 0:
                    self.gs(v).deploy_in_user(0, 0)
                else:
                    # t_i = self.get_input_ready(None, v, True)
                    t_i = self.get_input_ready(v, self.pos_user)
                    self.gs(v).deploy_in_user(t_i, t_i)
                self.G_.add_edge(v, self.G_end)    
            else:
                # 根据EST调度
                self.edge_server_selection(v, func_edge_download, func_prepare, func_process)
                
                # 根据cloud clone
                # t_i_c = self.get_input_ready_for_cloud(v)
                t_i_c = self.get_input_ready(v, self.pos_cloud)
                self.gs(v).deploy_in_cloud(t_i_c, t_i_c + func_process[v][-1])
                # TODO. 如何管理cloud的资源
                self.G_.add_edge(v, self.G_end)
                self.G_.add_edge(-v, self.G_end)

            # update
            for s in G.successors(v):
                L.put((-priority_dict[s],s))
            # print(v)
            self.is_scheduler.add(v)
            self.task_refinement(self.G_, self.G, v)

        for s in self.strategy:
            s.debug_readable()
        # draw_dag(self.G_)


if __name__ == '__main__':
    data = Data("./data/data_2.txt")
    # draw_dag(data.G)
    SDTS(data).sdts()