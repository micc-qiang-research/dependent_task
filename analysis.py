from cluster import Core,Cluster
import numpy as np
import math
from util import *
from scheduler.executor import Executor
from strategy import Strategy

class Analysis(Executor):
    def __init__(self, data, replica, place, download_sequence):
        super().__init__(data)
        self.data = data
        self.replica = replica
        self.place = place
        self.download_sequence = download_sequence

        self.cloud_start_core_number = self.cluster.get_total_core_number() - self.servers[-1].core

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