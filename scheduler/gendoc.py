import networkx as nx
from .scheduler import Scheduler
import math
import numpy as np
from .executor import Executor

class GenDoc(Scheduler):
    def __init__(self, data, config):
        super().__init__(data, config)
        # edge server - func config deploy
        self.fc_es_deploy = np.array([[0 for _ in range(self.K)] for _ in range(self.N)])
        
        # 假设所有server部署了所有函数
        # for i in range(self.K):
        for j in range(self.source+1, self.sink):
            self.deploy_func_config(self.get_cloud_id(),j)

    def get_func_config_size_by_server_id(self, server_id):
        return np.sum(self.fc_es_deploy[:,server_id])


    def deploy_func_config(self,server, func):
        assert server < self.K, "server error"
        assert func != self.source and func != self.sink, "source and sink can not be deploy in server"
        self.fc_es_deploy[func][server] = 1

    def deploy_funcs_config(self,server, funcs):
        for func in funcs:
            self.deploy_func_config(server, func)

    # 决定每台edge server上部署什么函数
    def gendoc(self):
        edge_server_number = self.K - 1

        func_process_wo_cloud = self.func_process[1:self.sink,:-1].copy()

        '''
          对每一行，排序返回下标
          [[1,2,3]   ->  [[0,1,2]]
           [3,2,1]]       [2,1,0]
        '''
        func_process_rank = np.argsort(func_process_wo_cloud, axis=1)
        func_process_rank_idx = [0] * func_process_rank.shape[0]
        # 获取一列，每一列的一个元素代表此函数在哪个server上运行的时间最短
        col = func_process_rank[:,0]

        # 获取最多次数
        C_ = -1
        for i in range(edge_server_number):
            funcs = []
            for j in col:
                if j == i:
                    funcs.append(i)
            C_ = max(C_, self.get_func_total_size(funcs))

        C_vir = [max(C_, c.storage) for c in self.servers[:-1]]

        assert len(C_vir) == edge_server_number, "edge server number not match"

        S_ = set(range(edge_server_number))
        while len(S_) != 0:
            ok = False
            for func_id in range(1, self.sink): # all func configure
                rank_idx = func_process_rank_idx[func_id-1]
                server_id = func_process_rank[func_id-1][rank_idx]
                func_process_rank_idx[func_id-1] += 1

                # deploy func config
                if server_id in S_:
                    ok = True
                    self.deploy_func_config(server_id, func_id)

                    # reach capacity limit
                    if not self.func_deploy_and_check(server_id, func_id, C_vir[server_id]):
                        S_.remove(server_id)
            # no process
            if not ok:
                break
        
        print("func config: \n", self.fc_es_deploy)

        return self.fixdoc()
    
    ############## fixdoc start ######################

    '''
        func 部署到 server上完成时间，若server为None，则部署到生成该请求的server上
        u -> v      (func_id)
        k    server (server_id)
        前置执行完时间 + 数据传输时间 + 执行时间
    '''
    def get_earliest_finish(self, v, server, P, F):
        if v == self.source:
            return 0
        res = 0
        strategy = {}
        for u in self.G.predecessors(v):
            # 根据每个前置任务，计算最短完成时间
            predence = math.inf
            if u == self.source:
                assert server!=None, "source trans data to sink is not allow"
                ans = self.get_weight(u,v)*self.server_comm[self.generate_pos][server] + P[v][server]
                predence = ans
                strategy[u] = -1 # 代表user
            else:
                for k in range(self.K):
                    if server is None: # sink
                        ans = F[u][k] + self.get_weight(u,v)*self.server_comm[k][self.generate_pos]
                    else:
                        ans = F[u][k] + self.get_weight(u,v)*self.server_comm[k][server] + P[v][server]
                    if predence > ans:
                        predence = ans
                        strategy[u] = k
            res = max(res, predence)
        return res, strategy
    
    '''
    Input:
        strategy_dict: 指明了各个节点达到最优时，前置节点部署的位置
        
        如 (i,k):{func_id1: server_id1, func_id2, server_id2...}，指明i函数部署到服务器k达到最优时前置节点部署的位置为：前置func_id1->server_id1; func_id2->server_id2

    Output:
        [(func_id1, server_id1), (func_id2, server_id2)...]: 指明达到最优需要部署的函数及其位置
    '''
    def fixdoc_strategy_parse(self, strategy_dict):
        sink_predcessor = strategy_dict[(self.sink,self.generate_pos)]
        place_strategy = [(k, sink_predcessor[k]) for k in sink_predcessor]
        res = set()
        res.add((self.source, self.generate_pos))
        res.add((self.sink, self.generate_pos))
        while len(place_strategy) > 0:
            strategy = place_strategy.pop()
            if strategy in res:
                continue
            if strategy[0] == self.source:
                continue
            res.add(strategy)
            s = strategy_dict[strategy]
            place_strategy.extend([(k, s[k]) for k in s])
        print(res)
        self.sched = res
        return res

    # 在资源固定好后（即每台server可部署哪些函数已经确定），决策每个func部署到哪个server
    def fixdoc(self):
        nodes = list(nx.topological_sort(self.G))
        
        # func在server上的最早开始时间
        F = [[math.inf for _ in range (self.K)] for _ in range(self.N)]

        # 若server上有func的环境，为0，否则为无穷
        P = self.func_process.copy()
        for i in nodes:
            for k in range(self.K):
                if self.fc_es_deploy[i][k] == 0:
                    P[i][k] = math.inf

        assert nodes[0] == 0, "source error"
        assert nodes[self.sink] == self.N-1, "sink error"
        strategy_dict = {}
        for i in nodes[1:-1]:
            for k in range(self.K):
                F[i][k],strategy = self.get_earliest_finish(i, k, P, F)
                strategy_dict[(i,k)] = strategy
        res, strategy = self.get_earliest_finish(self.sink, None,P, F)
        strategy_dict[(self.sink, self.generate_pos)] = strategy
        print(res, strategy)
        return self.fixdoc_strategy_parse(strategy_dict)

    ###### fixdoc end ####################################

    '''
      转换为特定策略，做了如下妥协：
      1. 原策略没有考虑任务并行性，或者说没有考虑一台机器同时执行任务的限制。这里对分配到某台机器上的所有任务按核平均分配
    '''
    def output_scheduler_strategy(self):
        replica = True
        place = [[] for i in range(self.cluster.get_total_core_number())]
        download_sequence = None
        core_index = [0 for i in range(self.K)] # 计算当前正在使用server的core index

        for func,server in self.sched:
            place[core_index[server]].append(func)
            core_index[server] += 1
            core_index[server] %= self.cluster.get_core_number(server)
        # place = [[0,1],[3],[1,2],[2]]
        return replica, place, download_sequence, Executor.TOPOLOGY

    def schedule(self):
        # self.fixdoc()
        self.gendoc()
        return self.output_scheduler_strategy()
        