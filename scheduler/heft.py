from .scheduler import Scheduler
from heft import heft, gantt
import numpy as np

class HEFT(Scheduler):
    def __init__(self, data):
        self.data = data
        self.func_process = np.array(self.data.func_process)
        self.server_comm = np.array(self.data.server_comm)

        # 每个核作为一个proc，需要改写server_comm矩阵，假设每个edge server有两个核，cloud有|func|个核，即不限数量

        self.G = self.data.G

    def schedule(self):
        sched, _, _ = heft.schedule_dag(self.G, 
                            communication_matrix=self.server_comm, 
                            computation_matrix=self.func_process,communication_startup=np.zeros(self.server_comm.shape[0]))
        gantt.showGanttChart(sched)