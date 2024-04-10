import numpy as np
from typing import NamedTuple
from docplex.mp.model import Model
import networkx as nx
from .scheduler import Scheduler
from .sdtsPlus import SDTSPlus
from .executor import Executor
import cvxpy as cp

class Propose(Scheduler):
    def is_func_has_layer(self, func, layer):
        if layer in self.funcs[func].layer:
            return 1
        return 0

    def get_func_layers(self, func_ids):
        s = set()
        for i in func_ids:
            s = s.union(set(self.funcs[i].layer))
        return list(s)

    def solve_deploy_model(self):
        mdl = Model(name="propose_solver")
        funcs = self.funcs
        servers = self.servers
        layers = self.layers
        
        range_func = range(self.N)
        range_server = range(self.K)
        range_layer = range(self.L)
        
        self.max_core_number = max([s.core for s in servers])
        range_core = range(self.max_core_number)

        self.range_func,self.range_server,self.range_layer,self.range_core = range_func,range_server,range_layer,range_core

        ############### 1.关键决策变量 #############
        # h_i_n_k，函数i是否在机器n的核k上运行
        mdl.h = mdl.continuous_var_cube(range_func, range_server, range_core, name=lambda fsc: "h_%d_%d_%d" % fsc)
        
        # ############# 2.辅助决策变量 ###############
        # X_i_n 函数i是否部署到机器n
        mdl.X = mdl.continuous_var_matrix(range_func, range_server, name=lambda fs: "X_%d_%d" % fs)

        # XX_i_j_n_n' 函数i是否部署到 n，并且函数j是否部署到 n'
        XX_ = [(i,j,n,n_) for i in range_func for j in range_func for n in range_server for n_ in range_func]
        mdl.XX = mdl.continuous_var_dict(XX_, name=lambda ffnn: "XX_%d_%d_%d_%d" % ffnn, ub=1)

        # H_i_j 函数i和j是否部署到同一台机器的同一个核
        mdl.H = mdl.continuous_var_matrix(range_func, range_func, name=lambda ff: "H_%d_%d" % ff)

        # g_n_l 机器n是否下载镜像块l
        mdl.G = mdl.continuous_var_matrix(range_server, range_layer, name=lambda sl: "g_%d_%d" % sl, ub=1) # server n 下不下载 layer l

        '''
        y_i_j 函数i和函数j至少一个先执行        
        '''
        mdl.Y = mdl.continuous_var_matrix(range_func, range_func, name=lambda ff: "y_%d_%d" % ff)


        # 函数i和函数j之间的通信带宽
        mdl.B = mdl.continuous_var_matrix(range_func, range_func, name=lambda ff: "b_%d_%d" % ff)
        # 时间
        mdl.T_data = mdl.continuous_var_list(range_func, name=lambda l: "t_data_%d" % l) # 数据依赖准备好时间
        mdl.T_start = mdl.continuous_var_list(range_func, name=lambda l: "t_start_%d" % l) # 开始时间
        mdl.T_end = mdl.continuous_var_list(range_func, name=lambda l: "t_end_%d" % l) # 结束时间

        h,X,XX,B,H,T_data,T_start,T_end,G,Y = mdl.h,mdl.X,mdl.XX,mdl.B,mdl.H,mdl.T_data,mdl.T_start,mdl.T_end,mdl.G,mdl.Y

        M = 1e11 # 一个足够大的数

        ########## 决策 end ###########################

        # source和sink部署到固定的机器
        mdl.add_constraints(X[i,self.generate_pos] == 1 for i in [self.source, self.sink])
        mdl.add_constraints(h[i,self.generate_pos, 0] == 1 for i in [self.source, self.sink])

        #（1）每个函数只能部署在一个服务器上
        mdl.add_constraints(mdl.sum(X[i, n] for n in range_server) == 1 for i in range_func)

        # 计算带宽
        mdl.add_constraints(XX[i,j,n,n_] >= X[i,n]+X[j,n_]-1\
            for i in range_func \
                for j in range_func \
                    for n in range_server \
                        for n_ in range_server)
        
        mdl.add_constraints(XX[i,j,n,n_] >= 0\
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
        

        # （16） 开始时间与数据传输及镜像准备好时间的关系
        mdl.add_constraints(T_start[i] >= T_data[i] for i in range_func)

        #（4）结束时间
        mdl.add_constraints((T_end[i] == T_start[i] + \
                                mdl.sum(X[i,n] * self.func_process[i][n] for n in range_server)) \
                    for i in range_func)

        # 以下两条约束保证，server含有函数所需的镜像块且相同的镜像块至多含有一个
        mdl.add_constraints( G[n, l] <= mdl.sum(X[i,n]*self.is_func_has_layer(i, l)\
            for i in range_func) \
                for n in range_server 
                    for l in range_layer) 
        
        mdl.add_constraints( G[n, l] >= mdl.sum(X[i,n]*self.is_func_has_layer(i, l) for i in range_func)/M \
                for n in range_server 
                    for l in range_layer)
        
        # (21) 函数部署到机器的某个核上 
        mdl.add_constraints(mdl.sum(h[i, n, c] for c in range_core) == X[i, n] \
                        for i in range_func \
                        for n in range_server) 
        
        # 若某台机器没有那么多核，则强制h[i,n,c]=0
        for n in range_server:
            if servers[n].core < self.max_core_number:
                mdl.add_constraints(h[i, n, k] == 0 \
                for i in range_func \
                    for n in range_server \
                        for k in range(servers[n].core, self.max_core_number))
        
        # (22) H[i,j]
        mdl.add_constraints(H[i,j] >= h[i,n,k] + h[j,n,k]-1\
                        for i in range_func \
                            for j in range_func \
                                for n in range_server \
                                    for k in range_core)
        

        # (23) 每个时间点运行的任务数量不超过机器核数
        mdl.add_constraints(T_end[j] - T_start[i] <= (2-Y[j,i]-H[i,j])*M \
                            for i in range_func \
                            for j in range_func if i!=j)
        
        mdl.add_constraints(T_end[i] - T_start[j] <= (2-Y[i,j]-H[i,j])*M \
                            for i in range_func \
                            for j in range_func if i!=j)
        
        # i在j前或j在i前至少有一个满足
        mdl.add_constraints(Y[i,j] + Y[j,i] >= 1 \
                            for i in range_func \
                            for j in range_func if i!=j) 
        
        mdl.minimize(mdl.sum([B[i,j] * self.get_weight(i,j) \
             for i in range_func for j in range_func if self.G.has_edge(i,j)]) + \
            mdl.sum([G[n,l] * self.layers[l].size * self.servers[n].download_latency for n in range_server for l in range_layer]))

        mdl.print_information()

        solution = mdl.solve()

        return mdl, solution

    def parse(self, mdl, solution):
        if solution:
            
            X = [[mdl.X[i,n].solution_value for n in self.range_server] for i in self.range_func]

            self.logger.debug(X)           

            exit(0)
            # self.random_rounding(h,P,X)
        else:
            self.logger.debug("求解失败")
            exit(1)
    

    def output_scheduler_strategy(self):
        pass

    def __init__(self, data, config):
        super().__init__(data, config)

    def schedule(self):
        mdl, solution = self.solve_deploy_model()
        self.parse(mdl, solution)
        exit(0)
        # mdl, solution = self.solve()
        # self.parse(mdl, solution)
        # return self.output_scheduler_strategy()
