from abc import abstractmethod, ABCMeta
import numpy as np
from strategy import SchedStrategy
from cluster import Cluster
import math
from util import *

class Scheduler(metaclass=ABCMeta):

    def __init__(self,data,config):
        self.data = data
        self.config = config

        # Input
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
        self.func_startup = np.array(self.data.func_startup) # 函数环境大小

        self.strategy = [SchedStrategy(i, self.data.N) for i in range(self.data.N)] # 任务放置的服务器、核、开始时间、结束时间

        self.cluster = Cluster([i.core for i in self.servers])

    ########## deploy strategy 得到某个func的部署edge位置
    # function strategy
    def gs(self, func):
        return self.strategy[func]

    ###################################################

    # 得到边的权重
    def get_weight(self, s, d):
        res = self.G.edges[s,d]["weight"]
        return res
    
    def get_cloud_id(self):
        return self.K-1
    
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
    
    def trans_strategy(self, strategy):
        

        for func,procs in strategy:
            for proc in procs:
                if func == self.source:
                    self.gs(func).deploy_in_user(0, 0)
                elif func == self.sink:
                    t_i = self.get_input_ready(func, self.pos_user, False)
                    self.gs(func).deploy_in_user(t_i, t_i)
                elif proc >= self.cluster.get_total_core_number():
                    server_id = self.get_cloud_id()
                    t_start = self.get_input_ready(func, self.pos_cloud, False)
                    t_end = t_start + self.func_process[func][server_id]
                    self.gs(func).deploy_in_cloud(t_start, t_end)
                else:
                    server_id, core_id = self.cluster.get_server_by_core_id(proc)

                    # 环境准备好时间
                    t_e = self.cluster.get_download_complete(server_id) + self.func_edge_download[func][server_id] + self.func_prepare[func]
                    
                    # 数据依赖准备好时间
                    self.gs(func).deploy_in_edge(server_id) # 模拟
                    t_i = self.get_input_ready(func, self.pos_edge, False)

                    t = max(t_e, t_i)
                    # 函数开始执行时间
                    t_execute_start = self.cluster.get_core_EST(server_id, core_id, self.func_prepare[func], self.func_process[func][server_id],t)

                    # 得到其他的衍生信息
                    t_execute_end = t_execute_start + self.func_process[func][server_id]

                    t_download_start = self.cluster.get_download_complete(server_id)
                    t_download_end = t_download_start + self.func_edge_download[func][server_id]
                    self.cluster.set_download_complete(server_id, t_download_end)

                    t_prepare_start = t_execute_start - self.func_prepare[func]
                    t_prepare_end = t_execute_start

                    self.strategy[func].deploy_in_edge(server_id, core_id, \
                        t_download_start=t_download_start, \
                        t_download_end=t_download_end, \
                        t_prepare_start=t_prepare_start, \
                        t_prepare_end=t_prepare_end,\
                        t_execute_start=t_execute_start, \
                        t_execute_end=t_execute_end)
                    self.cluster.place(server_id, self.cluster.get_edge_server_core(server_id, core_id), t_prepare_start, t_execute_end)

    @abstractmethod
    def schedule(self):
        pass

    def showGantt(self, name):
        bars = ""
        for s in self.strategy:
            bars = bars + s.debug_readable(self.cluster)
        output_gantt_json(name, self.cluster.get_names(), bars[:-1], self.gs(self.sink).get_user_end())
        draw_gantt()

    def get_total_time(self):
        return self.gs(self.sink).get_edge_end()

    def show_result(self, name):
        print("total time: {:.2f}".format(self.get_total_time()))
        if self.config.gantta:
            self.showGantt(name)