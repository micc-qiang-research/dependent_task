import networkx as nx
from .scheduler import Scheduler
import math

class GenDoc(Scheduler):
    def __init__(self, data, config):
        super().__init__(data, config)


        # 假设有10个函数，0、9是source和sink
        # 有4个server：3个edge，1个cloud

        # edge server - func config deploy
        self.fc_es_deploy = [[0 for _ in range(self.K)] for _ in range(self.N)]
        self.deploy_funcs_config(0,[1,2,3,4,5,6,7,8])
        self.deploy_funcs_config(1,[3,4,5,6])
        self.deploy_funcs_config(2,[5,6,7,8])
        self.deploy_funcs_config(3, range(self.source+1, self.sink))


    def deploy_func_config(self,server, func):
        assert server < self.K, "server error"
        assert func != self.source and func != self.sink, "source and sink can not be deploy in server"
        self.fc_es_deploy[func][server] = 1

    def deploy_funcs_config(self,server, funcs):
        for func in funcs:
            self.deploy_func_config(server, func)


    # func 部署到 server上完成时间，若server为None，则部署到user device
    # u -> v      (func_id)
    # k    server (server_id)
    # 前置执行完时间 + 数据传输时间 + 执行时间
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
                ans = self.get_weight(u,v)*self.get_comm(-1,server) + P[v][server]
                predence = ans
                strategy[u] = -1 # 代表user
            else:
                for k in range(self.K):
                    if server is None: # sink
                        ans = F[u][k] + self.get_weight(u,v)*self.get_comm(k,-1)
                    else:
                        ans = F[u][k] + self.get_weight(u,v)*self.get_comm(k ,server) + P[v][server]
                    if predence > ans:
                        predence = ans
                        strategy[u] = k
            res = max(res, predence)
        return res, strategy
    
    '''
      转换为特定策略，做了如下妥协：
      1. 原策略每个func可以部署到多个edge server，这里只选择第一个出现的edge server部署，但是在cloud可以同时在edge和cloud进行部署
      2. 原策略没有考虑任务并行性，或者说没有考虑一台机器同时执行任务的限制。这里对分配到某台机器上的所有任务按核平均分配
    '''
    def trans_strategy(self, sched):
        raw_strategy = [[] for i in range(self.cluster.get_total_core_number()+1)]
        core_index = [0 for i in range(self.K)] # 计算当前正在使用server的core index
        func_is_deploy_in_edge = set() # 如果函数已经在edge中部署过了，就不能再部署到edge

        for func,server in sched:
                if server == self.K-1: # cloud
                    raw_strategy[-1].append(func)
                else: # edge
                    if func in func_is_deploy_in_edge:
                        continue
                    func_is_deploy_in_edge.add(func)
                    raw_strategy[core_index[server]].append(func)
                    core_index[server] += 1
                    core_index[server] %= self.cluster.get_core_number(server)
        print(raw_strategy)
                
        super().trans_strategy(self.topology_gen_strategy(raw_strategy))
    
    def fixdoc_strategy_parse(self, end, strategy_dict):
        place_strategy = [(k, end[k]) for k in end]
        res = set()
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
        return res


    # 在资源固定好后（即每台server可部署哪些函数已经确定）
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
        print(res, strategy)
        return self.fixdoc_strategy_parse(strategy, strategy_dict)



    def schedule(self):
        sched = self.fixdoc()
        self.trans_strategy(sched)

        self.show_result("GenDoc")
        