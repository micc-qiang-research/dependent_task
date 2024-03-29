from .scheduler import Scheduler
from heft import heft, gantt
import numpy as np
import networkx as nx
from util import *
from .executor import Executor

'''
将HEFT算法用在本场景：
方法：
    每个核相当于HEFT的一个processor，然后调用heft进行调度
    得到调度结果后，保存每个函数调度的位置（调度到哪个核）
    映射到实际场景，使用拓扑排序的顺序调度每个函数，上述算法只确定调度位置

局限性：
    没有考虑函数的启动延迟，偏好将函数调度到相同机器减少数据传输开销，使得某机器拉取镜像时间过长
'''

class HEFT(Scheduler):
    def __init__(self, data,config):
        super().__init__(data, config)

        # 每个核作为一个proc，需要改写server_comm矩阵，假设每个edge server有两个核，cloud有|func|个核，即不限数量
        # TODO. 自定义核的数量
        self.func_number = self.N
        self.cores = np.array([s.core for s in self.servers], dtype=int)
        self.proc_number = np.sum(self.cores)
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


    def extend(self, arr):
        res = []
        for idx, ele in enumerate(arr):
            for core in range(self.cores[idx]):
                res.append(ele)
        return res

    def extend_process(self):
        total_process = []
        for process in self.total_process:
            row = self.extend(process)
            total_process.append(row)
        total_process = np.array(total_process)
            
        assert total_process.shape[1] == self.proc_number, "total_process error"
        return total_process

    def extend_comm(self):
        server_comm = []
        for idx,comm in enumerate(self.server_comm):
            row = self.extend(comm)
            for core in range(self.cores[idx]):
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

    def output_scheduler_strategy(self):
        sched = self.sched
        replica = False
        place = [[] for i in range(self.proc_number)]
        download_sequence = None
        
        for server in sched:
            for scheduler in sched[server]:
                task = scheduler.task
                proc = scheduler.proc
                place[proc].append(task)

        return replica, place, download_sequence, Executor.CUSTOM, self.sorted_nodes


    def change_comm2band(self):
        for i in range(self.proc_number):
            for j in range(self.proc_number):
                self.server_comm[i][j] = 1/self.server_comm[i][j] if self.server_comm[i][j] != 0 else self.server_comm[i][j]

    def schedule(self):
        self.change_comm2band() # heft需要使用带宽
        self.sorted_nodes ,self.sched, self.task_sched, _ = heft.schedule_dag(self.G, 
                            communication_matrix=self.server_comm, 
                            computation_matrix=self.total_process,communication_startup=np.zeros(self.server_comm.shape[0]))
        self.logger.debug(self.sched)
        self.logger.debug(self.task_sched)
        
        # 根据开始时间决定调度顺序
        # sequence_order = [item[0] for item in sorted(self.task_sched.items(), key=lambda p:p[1].start)]
        # self.logger.debug(sequence_order)
        
        return self.output_scheduler_strategy()

        # gantt.showGanttChart(sched)
        # self.trans_to_strategy(task_sched)
        # for s in self.strategy:
        #     s.debug()
        # self.show_result("heft")
