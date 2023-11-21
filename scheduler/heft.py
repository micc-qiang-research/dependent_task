from .scheduler import Scheduler
from heft import heft, gantt
import numpy as np
from strategy import SchedStrategy
import networkx as nx
from util import *

class HEFT(Scheduler):
    def __init__(self, data):
        super().__init__(data)

        # 每个核作为一个proc，需要改写server_comm矩阵，假设每个edge server有两个核，cloud有|func|个核，即不限数量
        # TODO. 自定义核的数量
        self.func_number = self.N
        self.proc_number = (self.server_comm.shape[0] - 1) * 2 + self.func_number

        self.func_process = self.extend_process()
        self.server_comm = self.extend_comm()

    def extend(self, comm, n):
        res = []
        for i in comm[:-1]:
            res.append(i)
            res.append(i)
        for i in range(n):
            res.append(comm[-1])
        return res

    def extend_process(self):
        func_process = []
        for process in self.func_process:
            row = self.extend(process, self.func_number)
            func_process.append(row)
        func_process = np.array(func_process)
            
        assert func_process.shape[1] == self.proc_number, "func_process error"
        return func_process

    def extend_comm(self):
        server_comm = []
        for comm in self.server_comm[:-1]:
            row = self.extend(comm, self.func_number)
            server_comm.append(row)
            server_comm.append(row)
        row = self.extend(self.server_comm[-1], self.func_number)
        for i in range(self.func_number):
            server_comm.append(row)
        server_comm = np.array(server_comm)
        
        assert server_comm.shape[0] == self.proc_number, "server_comm error"
        return server_comm

    def get_server(self, core_id):
        if self.is_deploy_in_cloud(core_id):
            return self.data.K - 1
        else:
            return core_id // 2

    def get_core(self, core_id):
        assert not self.is_deploy_in_cloud(core_id), "deploy in cloud"
        return core_id % 2

    def is_deploy_in_cloud(self, core_id):
        return core_id >= self.proc_number - self.func_number

    def trans_to_strategy(self, task_sched):
        vertices = list(nx.topological_sort(self.G))
        for func in vertices:
            print(task_sched[func])
            proc = task_sched[func].proc

            if func == self.source:
                self.gs(func).deploy_in_user(0, 0)
            elif func == self.sink:
                t_i = self.get_input_ready(func, self.pos_user, False)
                self.gs(func).deploy_in_user(t_i, t_i)
            elif self.is_deploy_in_cloud(proc):
                server_id = self.get_server(proc)
                t_start = self.get_input_ready(func, self.pos_cloud, False)
                t_end = t_start + self.func_process[func][server_id]
                self.gs(func).deploy_in_cloud(t_start, t_end)
            else:
                server_id = self.get_server(proc)

                # 环境准备好时间
                t_e = self.cluster.get_download_complete(server_id) + self.func_edge_download[func][server_id] + self.func_prepare[func]
                
                # 数据依赖准备好时间
                self.gs(func).deploy_in_edge(server_id) # 模拟
                t_i = self.get_input_ready(func, self.pos_edge, False)

                # 函数开始执行时间
                t_execute_start = max(t_e, t_i)

                # 得到其他的衍生信息
                t_execute_end = t_execute_start + self.func_process[func][server_id]

                t_download_start = self.cluster.get_download_complete(server_id)
                t_download_end = t_download_start + self.func_edge_download[func][server_id]
                self.cluster.set_download_complete(server_id, t_download_end)

                t_prepare_start = t_execute_start - self.func_prepare[func]
                t_prepare_end = t_execute_start
                core = self.get_core(task_sched[func].proc)

                self.strategy[func].deploy_in_edge(server_id, core, \
                    t_download_start=t_download_start, \
                    t_download_end=t_download_end, \
                    t_prepare_start=t_prepare_start, \
                    t_prepare_end=t_prepare_end,\
                    t_execute_start=t_execute_start, \
                    t_execute_end=t_execute_end)




    def schedule(self):
        sched, task_sched, _ = heft.schedule_dag(self.G, 
                            communication_matrix=self.server_comm, 
                            computation_matrix=self.func_process,communication_startup=np.zeros(self.server_comm.shape[0]))
        # gantt.showGanttChart(sched)
        self.trans_to_strategy(task_sched)
        # for s in self.strategy:
        #     s.debug()
        self.showGantt("heft")
