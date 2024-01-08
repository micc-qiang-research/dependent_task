import numpy as np
from typing import NamedTuple
from docplex.mp.model import Model
import networkx as nx
from .scheduler import Scheduler
from .executor import Executor

class Optim(Scheduler):
    def is_func_has_layer(self, func, layer):
        if layer in self.funcs[func].layer:
            return 1
        return 0

    def solve(self):
        mdl = Model(name="optim_solver")
        funcs = self.funcs
        servers = self.servers
        layers = self.layers
        
        range_func = range(self.N)
        range_server = range(self.K)
        range_layer = range(self.L)
        
        self.max_core_number = max([s.core for s in servers])
        range_core = range(self.max_core_number)

        self.range_func,self.range_server,self.range_layer,self.range_core = range_func,range_server,range_layer,range_core


        ########### 决策 start ###########################
        # 1.关键决策变量
        # 函数i是否部署到机器n的核c
        mdl.h = mdl.continuous_var_cube(range_func, range_server, range_core, name=lambda fsc: "h_%d_%d_%d" % fsc, ub=1)
        # 镜像层拉取的序列关系
        mdl.P = mdl.continuous_var_cube(range_server, range_layer, range_layer, name=lambda sll: "p_%d_%d_%d" % sll, ub=1)
        
        # 2.辅助决策变量
        # 函数i是否部署到机器n
        mdl.X = mdl.continuous_var_matrix(range_func, range_server, name=lambda fs: "X_%d_%d" % fs, ub=1)

        # 函数i是否部署到 n，并且函数j是否部署到 n_
        XX_ = [(i,j,n,n_) for i in range_func for j in range_func for n in range_server for n_ in range_func]
        mdl.XX = mdl.continuous_var_dict(XX_, name=lambda ffnn: "XX_%d_%d_%d_%d" % ffnn, ub=1)

        # 函数i和函数j之间的通信带宽
        mdl.B = mdl.continuous_var_matrix(range_func, range_func, name=lambda ff: "b_%d_%d" % ff)

        # 函数i和j是否部署到同一台机器的同一个核
        mdl.H = mdl.continuous_var_matrix(range_func, range_func, name=lambda ff: "H_%d_%d" % ff, ub=1)

        # 时间
        mdl.T_data = mdl.continuous_var_list(range_func, name=lambda l: "t_data_%d" % l) # 数据依赖准备好时间
        mdl.T_start = mdl.continuous_var_list(range_func, name=lambda l: "t_start_%d" % l) # 开始时间
        mdl.T_end = mdl.continuous_var_list(range_func, name=lambda l: "t_end_%d" % l) # 结束时间
        mdl.T_image = mdl.continuous_var_list(range_func, name=lambda l: "t_image_%d" % l) # 镜像准备好时间

        # 镜像块是否下载
        mdl.G = mdl.continuous_var_matrix(range_server, range_layer, name=lambda sl: "g_%d_%d" % sl, ub=1) # server n 下不下载 layer l


        '''
        线性化辅助变量，满足如下条件，Q才为1
        1. i部署到n上
        2. n需要下载l1,l2
        3. l1下载比l2先
        '''
        Q_ = [(n,i,l1,l2) for n in range_server for i in range_func for l1 in range_layer for l2 in range_layer]
        mdl.Q = mdl.continuous_var_dict(Q_, name=lambda nill: "q_%d_%d_%d_%d" % nill, ub=1)

        '''
        线性化辅助变量        
        '''
        mdl.Y = mdl.continuous_var_matrix(range_func, range_func, name=lambda ff: "y_%d_%d" % ff, ub=1)
        mdl.TET = mdl.continuous_var()

        h,P,X,XX,B,H,T_data,T_start,T_end,T_image,G,Q,Y,TET = mdl.h,mdl.P,mdl.X,mdl.XX,mdl.B,mdl.H,mdl.T_data,mdl.T_start,mdl.T_end,mdl.T_image,mdl.G,mdl.Q,mdl.Y,mdl.TET

        M = 1e7 # 一个足够大的数

        ########## 决策 end ###########################


        ########## 约束 start ###########################
        # 每个函数只能部署在一个服务器上
        mdl.add_constraints(mdl.sum(X[f, s] for s in range_server) == 1 for f in range_func)
        
        # 函数部署到机器的某个核上 
        mdl.add_constraints(mdl.sum(h[i, n, c] for c in range_core) == X[i, n] \
                        for i in range_func \
                        for n in range_server) 
        
        # 若某台机器没有那么多核，则强制h[i,n,c]=0
        for n in range_server:
            if servers[n].core < self.max_core_number:
                mdl.add_constraints(h[i, n, c] == 0 \
                for i in range_func \
                    for n in range_server \
                        for c in range(servers[n].core, self.max_core_number))
        
        # 下面三条约束计算函数i和函数j之间的带宽
        mdl.add_constraints(XX[i,j,n,n_] >= (X[i,n]+X[j,n_]-1)/2\
            for i in range_func \
                for j in range_func \
                    for n in range_server \
                        for n_ in range_server)
        
        mdl.add_constraints(XX[i,j,n,n_] <= (X[i,n]+X[j,n_])/2\
            for i in range_func \
                for j in range_func \
                    for n in range_server \
                        for n_ in range_server)
        
        mdl.add_constraints(B[i,j] == mdl.sum( XX[i,j,n,n_]*self.server_comm[n][n_] for n_ in range_server for n in range_server) \
                for i in range_func \
                    for j in range_func)
    
        # 数据准备好时间
        mdl.add_constraints(\
            T_data[i] >= T_end[j] + B[j,i] * self.get_weight(j,i) \
                for j in range_func \
                    for i in range_func if self.G.has_edge(j,i))

        # 结束时间
        mdl.add_constraints((T_end[i] == T_start[i] + \
            mdl.sum(X[i,n] * self.func_process[i][n] 
                for n in range_server)) \
                    for i in range_func)

        # 函数只能调度到含有所需镜像块的机器
        mdl.add_constraints( X[i,n]*self.is_func_has_layer(i, l) <= G[n,l] \
            for i in range_func \
                for n in range_server \
                    for l in range_layer)

        # 以下两条约束保证，server含有函数所需的镜像块且相同的镜像块至多含有一个
        mdl.add_constraints( G[n, l] <= mdl.sum(X[i,n]*self.is_func_has_layer(i, l)\
            for i in range_func) \
                for n in range_server 
                    for l in range_layer) 
        
        mdl.add_constraints( G[n, l] >= mdl.sum(X[i,n]*self.is_func_has_layer(i, l) for i in range_func)/M \
                for n in range_server 
                    for l in range_layer)

        # 镜像下载大小不超过机器的存储
        mdl.add_constraints(mdl.sum(G[n,l] * layers[l].size for l in range_layer) <= servers[n].storage \
                            for n in range_server)

        mdl.add_constraints( P[n,l1, l2] + P[n,l2,l1] <= (G[n, l1] + G[n, l2]) / 2  \
                            for n in range_server \
                            for l1 in range_layer \
                            for l2 in range_layer if l1 != l2)

        mdl.add_constraints( P[n,l1, l2] + P[n,l2,l1] >= (G[n, l1] + G[n, l2] - 1) / 2  \
                            for n in range_server \
                            for l1 in range_layer \
                            for l2 in range_layer if l1 != l2)

        mdl.add_constraints( P[n,l, l] == G[n,l] \
                            for n in range_server \
                            for l in range_layer)

        mdl.add_constraints( P[n,l1, l2] + P[n, l2, l3] + P[n, l3, l1] <= 2 \
                            for n in range_server \
                            for l1 in range_layer\
                            for l2 in range_layer if l2 != l1 \
                            for l3 in range_layer if l3 != l2 and l3 != l1)

        mdl.add_constraints( Q[n,i,l1,l2] <= (X[i,n]+P[n,l1,l2])/2\
                            for n in range_server \
                            for i in range_func \
                            for l1 in range_layer \
                            for l2 in range_layer)

        mdl.add_constraints( Q[n,i,l1,l2] >= (X[i,n]+P[n,l1,l2]-1)/2\
                            for n in range_server \
                            for i in range_func \
                            for l1 in range_layer \
                            for l2 in range_layer)

        # 镜像块准备好的时间为所有层都准备好
        mdl.add_constraints( T_image[i] >= self.servers[n].download_latency * mdl.sum(Q[n,i,l_,l] * layers[l_].size for l_ in range_layer )\
                for n in range_server \
                    for l in range_layer \
                        for i in range_func if self.is_func_has_layer(i, l) == 1)

        mdl.add_constraints(T_start[i] >= T_data[i] for i in range_func)

        mdl.add_constraints(T_start[i] >= T_image[i] for i in range_func)

        ### H[i,j]为i仅当对某个n,k，h[i,n,k]和h[j,n,k]同时为1
        mdl.add_constraints(H[i,j] >= h[i,n,k] + h[j,n,k]-1\
                for n in range_server\
                    for k in range_core \
                        for i in range_func \
                            for j in range_func)
        
        mdl.add_constraints(H[i,j] <= 1 - (2 - h[i,n,k] - h[j,n,k])/M\
                for n in range_server\
                    for k in range_core \
                        for i in range_func \
                            for j in range_func)

        # 每个时间点运行的任务数量不超过机器核数
        mdl.add_constraints(T_end[j] - T_start[i] <= (2-Y[j,i]-H[i, j])*M \
                            for i in range_func \
                            for j in range_func if i!=j)
        
        mdl.add_constraints(T_end[i] - T_start[j] <= (2-Y[i,j]-H[i, j])*M \
                            for i in range_func \
                            for j in range_func if i!=j)
        
        mdl.add_constraints(Y[i,j] + Y[j,i] >= 1 \
                            for i in range_func \
                            for j in range_func if i!=j)
        
        mdl.add_constraints(TET >= T_end[i] for i in range_func)
        
        # 目标
        mdl.minimize(TET)
        mdl.print_information()

        solution = mdl.solve()

        return mdl, solution

    def get_server_core(self, processor):
        server_id = processor // self.max_core_number
        core_id = processor % self.max_core_number
        assert core_id < self.data.server_info[server_id][0]
        return server_id, core_id

    def random_rounding(self, h, P):
        layers = [set() for _ in self.range_server] # 每个服务器需要下载的镜像集合
        task_server_mapper = {}
        download_sequence = []
        for i,task in enumerate(h):
            probs = np.array(task).reshape(-1)
            processor = np.random.choice(len(probs), 1, p=probs)[0]
            server_id, core_id = self.get_server_core(processor)
            task_server_mapper[i] = (server_id, core_id)

            for layer in range(self.data.L):
                if layer in self.data.funcs[i].layer:
                    layers[server_id].add(layer)

        for i, server_p in enumerate(P):
            layers_priority = np.array(server_p).sum(axis=1)
            download_sequence.append(np.argsort(-layers_priority))

        self.logger.debug(task_server_mapper)
        self.logger.debug(download_sequence)
        self.task_server_mapper = task_server_mapper
        self.download_sequence = download_sequence
        # self.get_makespan(task_server_mapper, download_sequence)
            

    def parse(self, mdl, solution):
        if solution:
            # self.logger.debug(solution)
            h = []
            for i in self.range_func:
                h_ = [[mdl.h[i,n,c].solution_value for c in self.range_core] for n in self.range_server]
                self.logger.debug(f"h[{i}] = ")
                self.logger.debug(np.array(h_))
                h.append(h_)

            P = []
            for n in self.range_server:
                P_ = [[mdl.P[n,l1,l2].solution_value for l2 in self.range_layer] for l1 in self.range_layer]
                self.logger.debug(f"P[{n}] = ")
                self.logger.debug(np.array(P_))
                P.append(P_)

            self.random_rounding(h,P)
        else:
            self.logger.debug("求解失败")
            exit(1)

    def output_scheduler_strategy(self):
        replica = False
        place = [[] for i in range(self.cluster.get_total_core_number())]
        download_sequence = self.download_sequence
        for task_id in self.task_server_mapper:
            server_id, core_id = self.task_server_mapper[task_id]
            pid = self.cluster.get_total_core_id(server_id, core_id)
            place[pid].append(task_id)
        return replica, place, download_sequence, Executor.TOPOLOGY

    def __init__(self, data, config):
        super().__init__(data, config)

    def schedule(self):
        mdl, solution = self.solve()
        self.parse(mdl, solution)
        return self.output_scheduler_strategy()
