from abc import abstractmethod, ABCMeta
import numpy as np
from strategy import SchedStrategy
from cluster import Cluster
import math
from util import *

class Scheduler(metaclass=ABCMeta):

    def __init__(self,data):
        self.data = data
        self.G = self.data.G
        self.func_process = np.array(self.data.func_process) # 函数执行时间
        self.edge_bandwith = np.array(self.data.edge_bandwidth) # 边缘服务器带宽 1..(k-1)
        self.func_startup = np.array(self.data.func_startup) # 函数环境大小
        self.func_edge_download = self.get_func_edge_download(self.func_startup, self.edge_bandwith) # 函数在各个边缘下载时间
        self.func_prepare = np.array(self.data.func_prepare)
        self.server_comm = np.array(self.data.server_comm) # 服务器之间的通信时间
        self.ue_comm = np.array(self.data.ue_comm) # 用户<->edge
        self.uc_comm = self.data.uc_comm # 用户<->cloud

        self.N = self.data.N
        self.K = self.data.K
        self.source = 0
        self.sink = self.N - 1

        self.strategy = [SchedStrategy(i, self.data.N) for i in range(self.data.N)] # 任务放置的服务器、核、开始时间、结束时间

        self.cluster = Cluster(self.data.cores)

        self.pos_user = 0
        self.pos_edge = 1
        self.pos_cloud = 2

    ########## deploy strategy 得到某个func的部署edge位置
    # function strategy
    def gs(self, func):
        return self.strategy[func]

    ###################################################

    # 得到边的权重
    def get_weight(self, s, d):
        res = self.G.edges[s,d]["weight"]
        return res

    # 根据函数环境大小即边缘带宽，计算下载时间
    def get_func_edge_download(self, func_startup, edge_bandwith):
        return (np.tile(func_startup.reshape(len(func_startup), 1), (1, self.data.K-1)).T / edge_bandwith.reshape(-1,1)).T


    """
        func1部署在pos1，func2部署在pos2，计算func1的数据传输到func2的时间
        
        is_err表示pos1没有func1时是否报错
    """
    def input_ready(self, func1, func2, pos1, pos2, is_err=True):
        weight = self.get_weight(func1, func2)
        try:

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
        except Exception as e:
            if is_err:
                raise e
            else:
                return math.inf

    def get_input_ready(self, func2, pos2, is_err=True):
        res = 0
        for i in self.G.predecessors(func2):
            if i == self.source:
                v = self.input_ready(i, func2, self.pos_user, pos2, is_err)
            else:
                v = min(
                    self.input_ready(i, func2, self.pos_edge, pos2, is_err),
                    self.input_ready(i, func2, self.pos_cloud, pos2, is_err),
                )
            res = max(v, res)
        return res

    @abstractmethod
    def schedule(self):
        pass

    def showGantt(self, name):
        bars = ""
        for s in self.strategy:
            bars = bars + s.debug_readable(self.cluster)
        output_gantt_json(name, self.cluster.get_names(), bars[:-1], self.gs(self.sink).get_user_end())
        draw_gantt()