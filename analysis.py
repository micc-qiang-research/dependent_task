from cluster import Core,Cluster
import numpy as np
import math
from util import *

class Strategy:
    def __init__(self, func, N):
        self.func = func # 记录是哪个函数的策略
        self.N = N # 函数个数
        self.deployment = {}
    
    def __get_a_deployment(self,server_id, core_id=None, t_execute_start=None, t_execute_end=None,t_download_start=0, t_download_end=0):
        return {
            "valid": True,
            "server_id": server_id,
            "core_id": core_id,
            "t_execute_start": t_execute_start,
            "t_execute_end": t_execute_end,
            "t_download_start": t_download_start,
            "t_download_end": t_download_end
        }

    def is_deploy(self, name):
        if name not in self.deployment or not self.deployment[name]['valid']:
            return False
        return True
        
    def clear(self, name):
        if self.is_deploy(name):
            self.deployment[name]['valid'] = False

    def deploy(self, name, server_id, core_id=None, t_execute_start=None, t_execute_end=None,t_download_start=0, t_download_end=0):
        self.deployment[name] = self.__get_a_deployment(server_id, core_id, t_execute_start, t_execute_end, t_download_start, t_download_end)

    def get_deploy_info(self, name):
        assert self.is_deploy(name), "not deploy"
        return self.deployment[name]
    
    def get_all_deploys(self):
        return list(self.deployment.values())

    def get_name(self,server_id, core_id):
        return "server_" + str(server_id) + "_" + str(core_id)

    def get_download_name(self, server_id):
        return "server_" + str(server_id) + "_d"

    def debug_readable(self):
        from util import colors, prepare_color, download_color, user_color
        func = self.func
        if func != 0 and func != self.N - 1:
            if func - 1 >= len(colors)-3:
                assert False, "too many function"
            else:
                func_color = colors[func - 1]
        else:
            func_color = user_color
        bars = ""

        str_json = "{{\"row\": \"{}\", \"from\": {}, \"to\": {}, \"color\": \"{}\"}},"
        for deploy in self.get_all_deploys():
            name = self.get_name(deploy["server_id"], deploy["core_id"])
            name_download = self.get_download_name(deploy["server_id"])
            # download 
            bars = bars + str_json.format(name_download, deploy["t_download_start"], deploy["t_download_end"], download_color)
            
            # exec
            bars = bars + str_json.format(name, deploy["t_execute_start"], deploy["t_execute_end"], func_color)
        
        return bars

class Analysis:

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

    def gs(self, func):
        return self.strategy[func]
    
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
            if i == self.source:
                v = self.input_ready(i, func2, "edge", pos2)
            else:
                v = min(
                    self.input_ready(i, func2, "edge", pos2),
                    self.input_ready(i, func2, "cloud", pos2),
                )
            res = max(v, res)
        self.gs(func2).clear(pos2)
        return res
    
    def get_func_download_time(self, func_startup, edge_bandwith):
        return (np.tile(func_startup.reshape(len(func_startup), 1), (1, self.data.K)).T * edge_bandwith.reshape(-1,1)).T
    
    def get_makespan(self):
        return self.gs(self.sink).get_deploy_info("edge")["t_execute_end"]
    
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
        self.func_startup = np.array(self.data.func_startup) # 函数环境大小

    def __init__(self, data, replica, place, download_sequence):
        self.data = data
        self.replica = replica
        self.place = place
        self.download_sequence = download_sequence

        self.read_info()
        self.cluster = Cluster([i.core for i in self.data.servers])
        self.strategy = [Strategy(i,self.data.N) for i in range(self.N)]
        self.cloud_start_core_number = self.cluster.get_total_core_number() - self.servers[-1].core

        self.func_download_time = self.get_func_download_time(self.func_startup, np.array([s.download_latency for s in self.servers]))

        self.execute()
        print(self.get_makespan())
        self.showGantt("SDTS")

    def showGantt(self, name):
        bars = ""
        for s in self.strategy:
            bars = bars + s.debug_readable()
        output_gantt_json(name, self.cluster.get_names(), bars[:-1], self.get_makespan())
        draw_gantt()        

    def execute(self):
        for func,procs in self.dumb_gen_strategy(self.place):
            for proc in procs:
                if func == self.source:
                    self.gs(func).deploy("edge", self.generate_pos, 0,0,0,0,0)
                elif func == self.sink:
                    t_i = self.get_input_ready(func, "edge", self.generate_pos)
                    self.gs(func).deploy("edge", self.generate_pos, 0, t_i, t_i, t_i, t_i)
                else:
                    pos = "edge"
                    if proc >= self.cloud_start_core_number:
                        pos = "cloud"
                    server_id, core_id = self.cluster.get_server_by_core_id(proc)

                    # 环境准备好时间
                    t_e = self.cluster.get_download_complete(server_id) + self.func_download_time[func][server_id]

                    # 数据依赖准备好时间
                    t_i = self.get_input_ready(func, pos, server_id)

                    t = max(t_e, t_i)
                    # 函数开始执行时间
                    t_execute_start = self.cluster.get_core_EST(server_id, core_id, 0, self.func_process[func][server_id],t)

                    # 得到其他的衍生信息
                    t_execute_end = t_execute_start + self.func_process[func][server_id]

                    t_download_start = self.cluster.get_download_complete(server_id)
                    t_download_end = t_download_start + self.func_download_time[func][server_id]
                    self.cluster.set_download_complete(server_id, t_download_end)

                    self.strategy[func].deploy(pos, \
                        server_id=server_id, \
                        core_id=core_id, \
                        t_download_start=t_download_start, \
                        t_download_end=t_download_end, \
                        t_execute_start=t_execute_start, \
                        t_execute_end=t_execute_end)
                    
                    self.cluster.place(server_id, core_id, t_execute_start, t_execute_end)