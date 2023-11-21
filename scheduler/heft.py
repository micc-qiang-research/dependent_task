from .scheduler import Scheduler
from heft import heft, gantt
import numpy as np

class HEFT(Scheduler):
    def __init__(self, data):
        self.data = data
        func_process = np.array(self.data.func_process)
        server_comm = np.array(self.data.server_comm)

        # 每个核作为一个proc，需要改写server_comm矩阵，假设每个edge server有两个核，cloud有|func|个核，即不限数量
        # TODO. 自定义核的数量
        self.func_number = func_process.shape[0]
        self.proc_number = (server_comm.shape[0] - 1) * 2 + self.func_number

        self.extend_process(func_process)
        self.extend_comm(server_comm)

        self.G = self.data.G

    def extend(self, comm, n):
        res = []
        for i in comm[:-1]:
            res.append(i)
            res.append(i)
        for i in range(n):
            res.append(comm[-1])
        return res

    def extend_process(self, func_process):
        self.func_process = []
        for process in func_process:
            row = self.extend(process, self.func_number)
            self.func_process.append(row)
        self.func_process = np.array(self.func_process)
            
        assert self.func_process.shape[1] == self.proc_number, "func_process error"

    def extend_comm(self, server_comm):
        self.server_comm = []
        for comm in server_comm[:-1]:
            row = self.extend(comm, self.func_number)
            self.server_comm.append(row)
            self.server_comm.append(row)
        row = self.extend(server_comm[-1], self.func_number)
        for i in range(self.func_number):
            self.server_comm.append(row)
        self.server_comm = np.array(self.server_comm)
        
        assert self.server_comm.shape[0] == self.proc_number, "server_comm error"


    def schedule(self):
        sched, _, _ = heft.schedule_dag(self.G, 
                            communication_matrix=self.server_comm, 
                            computation_matrix=self.func_process,communication_startup=np.zeros(self.server_comm.shape[0]))
        gantt.showGanttChart(sched)