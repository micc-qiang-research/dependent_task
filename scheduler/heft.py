from .scheduler import Scheduler
from heft import heft, gantt
import numpy as np

class HEFT(Scheduler):
    def __init__(self, data):
        self.data = data
        self.func_process = np.array(self.data.func_process)
        self.server_comm = np.array(self.data.server_comm)
        self.G = self.data.G

    def schedule(self):
        sched, _, _ = heft.schedule_dag(self.G, 
                            communication_matrix=self.server_comm, 
                            computation_matrix=self.func_process)
        gantt.showGanttChart(sched)