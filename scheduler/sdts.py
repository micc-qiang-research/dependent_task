import networkx as nx
import numpy as np
import portion as P
from util import *
from .scheduler import Scheduler
import math
from cluster import Core
from .executor import Executor
from .sequencing import GenStrategy

class SDTS(Scheduler):
    def __init__(self, data, config):
        super().__init__(data, config)
        self.G_ = nx.DiGraph()
        self.G_end = "end"
        
        self.func_edge_download = self.func_download_time[:,:-1] # sdts算法假设在cloud不需要环境准备时间
        
        self.is_scheduler = set() # 目前已经调度的节点

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
        self.logger.debug(priority_dict)
        return priority_dict

    def edge_server_selection(self, func, func_edge_download, func_prepare, func_process):
        early_start_time = P.inf
        early_core_id = None
        early_server_id = -1
        for idx, server in enumerate(self.cluster.get_server()[:-1]):
            t_e = self.cluster.get_download_complete(idx) + func_edge_download[func][idx] + func_prepare[func] # 环境准备好时间
            
            # t_i = self.get_input_ready(idx, func) # 数据依赖准备好时间
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
        t_download_finish = self.cluster.get_download_complete(early_server_id) + func_edge_download[func][early_server_id]
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

    def _update_G_(self, G_, source, dest):
        if source == self.source:
            assert dest != self.sink, "error"
            G_.add_edge(source, dest)
            G_.add_edge(source, -dest)
        elif dest == self.sink:
            if self.input_ready(source, dest, "edge", "edge") <= \
                self.input_ready(source, dest, "cloud", "edge"):
                G_.add_edge(source, dest)
            else:
                G_.add_edge(-source, dest)
        else:
            # dest
            if self.input_ready(source, dest, "edge", "edge") <= \
                self.input_ready(source, dest, "cloud", "edge"):
                G_.add_edge(source, dest)
            else:
                G_.add_edge(-source, dest)  

            # -dest
            if self.input_ready(source, dest, "edge", "cloud") <= \
                self.input_ready(source, dest, "cloud", "cloud"):
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
                    self.gs(-node).clear("cloud")
                    # TODO cloud的resource如何管理？
                else:
                    # release edge resource
                    
                    # 更新server的下载完成时间
                    server_id = self.gs(node).get_deploy_info("edge")["server_id"]
                    download_complete = self.cluster.get_download_complete(server_id)
                    self.cluster.set_download_complete(server_id, download_complete - self.func_edge_download[node][server_id])

                    # 释放占用的core的资源
                    edge_parms = self.gs(node).get_deploy_info("edge")
                    self.cluster.release(edge_parms["server_id"], edge_parms["core_id"], edge_parms["t_execute_start"]-0 , edge_parms["t_execute_end"])
                    
                    # 清除edge deployment
                    self.gs(node).clear("edge")

    def output_scheduler_strategy(self):
        total_core_number = self.cluster.get_total_core_number()
        replica = True # 该策略允许复制
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
            if strategy.is_deploy("cloud"):
                info = strategy.get_deploy_info("cloud")
                cloud_deploy.append({"func":j, "start": info["t_execute_start"], "end": info["t_execute_end"]})
        
        cloud_deploy = sorted(cloud_deploy, key=lambda x: x["start"])
        cloud_core_number = self.cluster.get_core_number(self.get_cloud_id())
        cloud_start_core_id = self.cluster.get_total_core_number() - cloud_core_number
        cloud_cores = [Core(i) for i in range(cloud_core_number)]
        for deploy in cloud_deploy:
            success = False
            for i, core in enumerate(cloud_cores):
                if core.is_occupy(deploy["start"], deploy["end"]):
                    continue
                else:
                    core.occupy(deploy["start"], deploy["end"])
                    sched_info[cloud_start_core_id + i].append({"func": deploy["func"], "start": deploy["start"]})
                    success = True
                    break
            
            # 若没有可以直接使用的核，选择可以最早开始的核
            if not success:
                est = 1e10
                choose = -1
                size = deploy["end"] - deploy["start"]
                for i, core in enumerate(cloud_cores):
                    tmp = core.find_est(deploy["start"], size)
                    if tmp < est:
                        est = tmp
                        choose = i
                cloud_cores[choose].occupy(est, size)
                sched_info[cloud_start_core_id + choose].append({"func": deploy["func"], "start": est})
                    
        for i in range(total_core_number):
            sched_info[i] = sorted(sched_info[i], key=lambda x: x["start"])
            for sched in sched_info[i]:
                place[i].append(sched["func"])

        self.logger.debug(f"replica? {replica}")
        self.logger.debug(f"place: {place}")
        self.logger.debug(f"download_sequence: {download_sequence}")
        
        return replica, place, download_sequence, GenStrategy.CUSTOM, self.sorted_nodes
            
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
        self.G_.add_node(self.G_end)
        
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
                self.G_.add_edge(v, self.G_end)
            else:
                # 根据EST调度
                self.edge_server_selection(v, func_edge_download, func_prepare, func_process)
                
                # 根据cloud clone
                # t_i_c = self.get_input_ready_for_cloud(v)
                t_i_c = self.get_input_ready(v, "cloud", self.get_cloud_id())
                self.gs(v).deploy("cloud", self.get_cloud_id(), 0, t_i_c, t_i_c + func_process[v][-1])
                # TODO. 如何管理cloud的资源
                self.G_.add_edge(v, self.G_end)
                self.G_.add_edge(-v, self.G_end)

            self.is_scheduler.add(v)

            # update
            for s in G.successors(v):
                if set(G.predecessors(s)).issubset(self.is_scheduler) and s not in self.is_scheduler:
                    L.put((-priority_dict[s],s))
            # print(v)
            self.task_refinement(self.G_, self.G, v)

        self.logger.debug(self.sorted_nodes)
        return self.output_scheduler_strategy()

        # draw_dag(self.G_)
        # self.show_result("sdts")
