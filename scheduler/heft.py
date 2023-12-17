from .scheduler import Scheduler
from heft import heft, gantt
import numpy as np
from strategy import SchedStrategy
import networkx as nx
from util import *

class HEFT(Scheduler):
    def __init__(self, data,config):
        super().__init__(data, config)

        # 每个核作为一个proc，需要改写server_comm矩阵，假设每个edge server有两个核，cloud有|func|个核，即不限数量
        # TODO. 自定义核的数量
        self.func_number = self.N
        self.cores = np.array(self.data.cores, dtype=int)
        self.proc_number = np.sum(self.cores) + self.func_number
        # self.proc_number = (self.server_comm.shape[0] - 1) * 2 + self.func_number

        self.total_process = self.get_total_process()
        self.total_process = self.extend_process()
        self.server_comm = self.extend_comm()

    def get_total_process(self):

        '''
        假设N=4，K=3
        [1,2,3,4] -> [[1 1 1]
                     [2 2 2]
                     [3 3 3]
                     [4 4 4]]
        
        '''
        prepare = np.tile(self.func_prepare, self.K).reshape(self.K, self.N).T
        return self.func_process + prepare


    def extend(self, comm, n):
        res = []
        for idx, com in enumerate(comm[:-1]):
            for core in range(self.cores[idx]):
                res.append(com)
        for i in range(n):
            res.append(comm[-1])
        return res

    def extend_process(self):
        total_process = []
        for process in self.total_process:
            row = self.extend(process, self.func_number)
            total_process.append(row)
        total_process = np.array(total_process)
            
        assert total_process.shape[1] == self.proc_number, "total_process error"
        return total_process

    def extend_comm(self):
        server_comm = []
        for idx,comm in enumerate(self.server_comm[:-1]):
            row = self.extend(comm, self.func_number)
            for core in range(self.cores[idx]):
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

    def trans_strategy(self, sched):
        strategy = [[] for i in range(self.proc_number)]
        for server in sched:
            for scheduler in sched[server]:
                task = scheduler.task
                proc = scheduler.proc
                strategy[proc].append(task)
        super().trans_strategy(strategy)

    def schedule(self):
        sched, task_sched, _ = heft.schedule_dag(self.G, 
                            communication_matrix=self.server_comm, 
                            computation_matrix=self.total_process,communication_startup=np.zeros(self.server_comm.shape[0]))
        # print(sched, task_sched)
        self.trans_strategy(sched)

        # gantt.showGanttChart(sched)
        # self.trans_to_strategy(task_sched)
        # for s in self.strategy:
        #     s.debug()
        self.show_result("heft")
