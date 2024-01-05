from cluster import Core,Cluster
import numpy as np
import math
from util import *
from scheduler.executor import Executor
from strategy import Strategy

class Analysis(Executor):
    def __init__(self, data, replica, place, download_sequence, gen_strategy = Executor.DUMB):
        super().__init__(data)
        self.data = data
        self.replica = replica
        self.place = place
        self.download_sequence = download_sequence
        self.gen_strategy = self.get_gen_strategy(gen_strategy)

        self.cloud_start_core_number = self.cluster.get_total_core_number() - self.servers[-1].core

        self.execute()
        print(self.get_makespan())
        self.showGantt("SDTS")

    # 返回开始下载时间和结束下载时间
    def download_layer(self, server_id, func_id):
        st = self.cluster.get_download_complete(server_id) # 起始下载位置
        # 指定seq
        if self.download_sequence:
            # 层下载完成时间
            if not hasattr(self, "layer_download_complete"):
                for seq in self.download_sequence:
                    self.layer_download_complete = [math.inf for i in range(self.L)]
                    download_time = 0
                    for layer in seq:
                        size = self.layers[layer].size
                        download_time += size * self.servers[server_id].download_latency
                        self.layer_download_complete[layer] = download_time
            st = 0
            res = -1
            for layer in self.funcs[func_id].layer:
                if self.layer_download_complete[layer] == math.inf:
                    assert False, f"server {server_id} need layer {layer}"
                if res < self.layer_download_complete[layer]:
                    res = self.layer_download_complete[layer]
        # 未指定seq
        else:
            if not hasattr(self, "layer_download_complete"):
                self.layer_download_complete = [math.inf for i in range(self.L)]
            
            res = -1
            for layer in self.funcs[func_id].layer:
                if self.layer_download_complete[layer] == math.inf:
                    self.cluster.set_download_complete(server_id, self.cluster.get_download_complete(server_id)+self.layers[layer].size * self.servers[server_id].download_latency)
                    self.layer_download_complete[layer] = self.cluster.get_download_complete(server_id)
                if res < self.layer_download_complete[layer]:
                    res = self.layer_download_complete[layer]
        return st, res

    def showGantt(self, name):
        bars = ""
        for s in self.strategy:
            bars = bars + s.debug_readable()
        output_gantt_json(name, self.cluster.get_names(), bars[:-1], self.get_makespan())
        draw_gantt()

    def execute(self):
        for func,procs in self.gen_strategy(self.place):
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
                    t_s, t_e = self.download_layer(server_id, func)

                    # 数据依赖准备好时间
                    t_i = self.get_input_ready(func, pos, server_id)

                    t = max(t_e, t_i)
                    # 函数开始执行时间
                    t_execute_start = self.cluster.get_core_EST(server_id, core_id, 0, self.func_process[func][server_id],t)

                    # 得到其他的衍生信息
                    t_execute_end = t_execute_start + self.func_process[func][server_id]


                    self.strategy[func].deploy(pos, \
                        server_id=server_id, \
                        core_id=core_id, \
                        t_download_start=t_s, \
                        t_download_end=t_e, \
                        t_execute_start=t_execute_start, \
                        t_execute_end=t_execute_end)
                    
                    self.cluster.place(server_id, core_id, t_execute_start, t_execute_end)