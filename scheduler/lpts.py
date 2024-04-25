import numpy as np
from typing import NamedTuple
from docplex.mp.model import Model
import networkx as nx
from .scheduler import Scheduler
from .executor import Executor
from .sdtsPlus import SDTSPlus
from heft import heft
import math

# linear program base task scheduling
class LPTS(SDTSPlus):
    def is_func_has_layer(self, func, layer):
        if layer in self.funcs[func].layer:
            return 1
        return 0

    def get_func_layers(self, func_ids):
        s = set()
        for i in func_ids:
            s = s.union(set(self.funcs[i].layer))
        return list(s)
    
    def iter_solve_deploy_model(self):
        min_val = 0
        max_val = 0
        for process in self.func_process:
            max_val += np.max(process)
        # 获取所需镜像块的总和
        total_layer_size = sum([self.layers[l].size for l in self.get_func_layers(range(self.N))])
        
        # 获取最大的拉取延迟
        max_fetch_latency = max([server.download_latency for server in self.servers])

        # makespan不可能超过这个时间
        #max_val = max(max_val, total_layer_size*max_fetch_latency+np.max(self.func_process))
        max_val += total_layer_size * max_fetch_latency

        iter_cnt = 0
        while(max_val - min_val > 1):
            TET = (min_val + max_val) / 2
            iter_cnt+=1
            self.logger.debug(f"--- iter {iter_cnt}: TET = {TET}---")
            __mdl, __solution = self.solve_deploy_model(TET)
            if __solution:
                max_val = TET # max_val保证了一定是可解的
                self.parse(__mdl, __solution)
            else:
                min_val = TET+1

        self.logger.debug(f"iter_cnt: {iter_cnt}")
        self.logger.debug(max_val)

    def build_deploy_model(self, TET):
        range_func = range(self.N)
        range_server = range(self.K)
        range_layer = range(self.L)

        if(hasattr(self, "mdl")):
            self.mdl.remove_constraints([f"TET{i}" for i in range_func])
            self.mdl.remove_constraints([f"TET_{n}" for n in range_server])
            self.mdl.add_constraints([self.mdl.T_end[i] <= TET for i in range_func], names=[f"TET{i}" for i in range_func])
            self.mdl.add_constraints(\
            [self.mdl.sum(self.mdl.G[n,l] * self.layers[l].size * self.servers[n].download_latency for l in range_layer) <= TET \
                for n in range_server], \
                    names=[f"TET_{n}" for n in range_server])
            return self.mdl
        
        self.mdl = Model(name="propose_solver")
        mdl = self.mdl
        
        self.max_core_number = max([s.core for s in self.servers])
        range_core = range(self.max_core_number)

        self.range_func,self.range_server,self.range_layer,self.range_core = range_func,range_server,range_layer,range_core
        
        # ############# 1. 决策变量 ###############
        # X_i_n 函数i是否部署到机器n
        mdl.X = mdl.continuous_var_matrix(range_func, range_server, name=lambda fs: "X_%d_%d" % fs, ub=1)

        # XX_i_j_n_n' 函数i是否部署到 n，并且函数j是否部署到 n'
        XX_ = [(i,j,n,n_) for i in range_func for j in range_func for n in range_server for n_ in range_func]
        mdl.XX = mdl.continuous_var_dict(XX_, name=lambda ffnn: "XX_%d_%d_%d_%d" % ffnn, ub=1)

        # g_n_l 机器n是否下载镜像块l
        mdl.G = mdl.continuous_var_matrix(range_server, range_layer, name=lambda sl: "g_%d_%d" % sl, ub=1) # server n 下不下载 layer l

        # 函数i和函数j之间的通信带宽
        mdl.B = mdl.continuous_var_matrix(range_func, range_func, name=lambda ff: "b_%d_%d" % ff)
        # 时间
        mdl.T_data = mdl.continuous_var_list(range_func, name=lambda l: "t_data_%d" % l) # 数据依赖准备好时间
        mdl.T_start = mdl.continuous_var_list(range_func, name=lambda l: "t_start_%d" % l) # 开始时间
        mdl.T_end = mdl.continuous_var_list(range_func, name=lambda l: "t_end_%d" % l) # 结束时间

        X,XX,B,T_data,T_start,T_end,G = mdl.X,mdl.XX,mdl.B,mdl.T_data,mdl.T_start,mdl.T_end,mdl.G

        mdl.comm_cost = mdl.continuous_var()
        mdl.fetch_cost = mdl.continuous_var()

        M = 1e9 # 一个足够大的数
        M_time = np.sum(self.func_process) / self.K
        fetch_weight = 0.5

        # layer的总大小作为g_n_l
        M_layer = [sum([1 for i in range_func if self.is_func_has_layer(i, l)]) for l in range_layer]

        # print(M_time, M_layer)

        ########## 决策 end ###########################

        # source和sink部署到固定的机器
        mdl.add_constraints(X[i,self.generate_pos] == 1 for i in [self.source, self.sink])

        #（1）每个函数只能部署在一个服务器上
        mdl.add_constraints(mdl.sum(X[i, n] for n in range_server) == 1 for i in range_func)

        # 计算带宽
        mdl.add_constraints(XX[i,j,n,n_] >= X[i,n]+X[j,n_]-1\
            for i in range_func \
                for j in range_func \
                    for n in range_server \
                        for n_ in range_server)
        
        mdl.add_constraints(B[i,j] == mdl.sum( XX[i,j,n,n_]*self.server_comm[n][n_] for n_ in range_server for n in range_server) \
                for i in range_func \
                    for j in range_func)

        # (30) 数据准备好时间：所有前置任务完成，并将数据发送
        mdl.add_constraints(\
            T_data[i] >= T_end[j] + B[j,i] * self.get_weight(j,i) \
                for j in range_func \
                    for i in range_func if self.G.has_edge(j,i))
        

        # （16） 开始时间与数据传输时间的关系
        mdl.add_constraints(T_start[i] >= T_data[i] for i in range_func)

        #（4）结束时间
        mdl.add_constraints((T_end[i] == T_start[i] + \
                                mdl.sum(X[i,n] * self.func_process[i][n] for n in range_server)) \
                    for i in range_func)

        # 以下两条约束保证，server含有函数所需的镜像块且相同的镜像块至多含有一个
        mdl.add_constraints( G[n, l] >= X[i,n]*self.is_func_has_layer(i, l)\
            for i in range_func \
                for n in range_server 
                    for l in range_layer) 
        
        # 函数结束时间小于TET
        mdl.add_constraints([T_end[i] <= TET for i in range_func], names=[f"TET{i}" for i in range_func])

        # 每台机器下载镜像的延迟不超过TET
        mdl.add_constraints(\
            [mdl.sum(G[n,l] * self.layers[l].size * self.servers[n].download_latency for l in range_layer) <= TET \
                for n in range_server], \
                    names=[f"TET_{n}" for n in range_server])

        mdl.add_constraint(mdl.comm_cost == mdl.sum([B[i,j] * self.get_weight(i,j) \
             for i in range_func for j in range_func if self.G.has_edge(i,j)]))
        
        mdl.add_constraint(mdl.fetch_cost == mdl.sum([G[n,l] * self.layers[l].size * self.servers[n].download_latency for n in range_server for l in range_layer]))

        # 目标是最小化总延迟
        mdl.minimize((1-fetch_weight)*mdl.comm_cost + fetch_weight*mdl.fetch_cost)

        mdl.print_information()
        return mdl


    def solve_deploy_model(self, TET = 1000):
        mdl = self.build_deploy_model(TET)
        solution = mdl.solve()
        return mdl, solution

    # 重写优先级方法，开始时间越小，优先级越高
    def priority(self, func_edge_download, server_comm, func_process):
        print(self.T_start)
        priority_dict = {k:-v for k,v in enumerate(self.T_start)}
        return priority_dict

    def parse(self, mdl, solution):
        if solution:
            # print(solution)
            print("comm_cost", mdl.comm_cost.solution_value)
            print("fetch_cost", mdl.fetch_cost.solution_value)
            X = np.array([[mdl.X[i,n].solution_value for n in self.range_server] for i in self.range_func])

            G = np.array([[mdl.G[n,l].solution_value for l in self.range_layer] for n in self.range_server])

            T_start = np.array([mdl.T_start[i].solution_value for i in self.range_func])

            # self.logger.debug(X)
            # self.logger.debug(G)
            self.deploy = np.argmax(X, axis=1)
            print(self.deploy)

            self.T_start = T_start

            # self.random_rounding(h,P,X)
        else:
            self.logger.debug("求解失败")
            exit(1)

    # def output_scheduler_strategy(self):
    #     replica = False
    #     place = [[] for i in range(self.cluster.get_total_core_number())]
    #     download_sequence = self.layer_sequence
    #     core_index = [0 for i in range(self.K)] # 计算当前正在使用server的core index

    #     for func,server in enumerate(self.deploy):
    #         place[core_index[server]+self.cluster.get_start_core_number(server)].append(func)
    #         core_index[server] += 1
    #         core_index[server] %= self.cluster.get_core_number(server)
    #     # place = [[0,1],[3],[1,2],[2]]
    #     return replica, place, download_sequence, Executor.TOPOLOGY

    def __init__(self, data, config):
        super().__init__(data, config)

    def schedule(self):
        # 生成self.deploy
        self.iter_solve_deploy_model()
        self.layer_sequence = None
        # self.priority(self.func_edge_download, self.server_comm, self.func_process)
        return super().schedule()
        # return self.output_scheduler_strategy()

        
