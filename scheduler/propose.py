import numpy as np
from typing import NamedTuple
from docplex.mp.model import Model
import networkx as nx
from .scheduler import Scheduler
from .sdtsPlus import SDTSPlus
from .executor import Executor

class Propose(SDTSPlus):
    def is_func_has_layer(self, func, layer):
        if layer in self.funcs[func].layer:
            return 1
        return 0

    def solve(self):
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

        use_binary = False
        image_related = False # 是否考虑Image相关的约束，若为False,假设下载镜像带宽无限，不用考虑镜像约束


        ########### 决策 start ###########################
        
        if use_binary:
            mdl.h = mdl.binary_var_cube(range_func, range_server, range_core, name=lambda fsc: "h_%d_%d_%d" % fsc)
            mdl.P = mdl.binary_var_cube(range_server, range_layer, range_layer, name=lambda sll: "p_%d_%d_%d" % sll)
            mdl.X = mdl.binary_var_matrix(range_func, range_server, name=lambda fs: "X_%d_%d" % fs)

            XX_ = [(i,j,n,n_) for i in range_func for j in range_func for n in range_server for n_ in range_func]
            mdl.XX = mdl.binary_var_dict(XX_, name=lambda ffnn: "XX_%d_%d_%d_%d" % ffnn, ub=1)

            mdl.H = mdl.binary_var_matrix(range_func, range_func, name=lambda ff: "H_%d_%d" % ff)
            
            mdl.G = mdl.binary_var_matrix(range_server, range_layer, name=lambda sl: "g_%d_%d" % sl) # server n 下不下载 layer l
            
            Q_ = [(n,i,l1,l2) for n in range_server for i in range_func for l1 in range_layer for l2 in range_layer]
            mdl.Q = mdl.binary_var_dict(Q_, name=lambda nill: "q_%d_%d_%d_%d" % nill)
            
            mdl.Y = mdl.binary_var_matrix(range_func, range_func, name=lambda ff: "y_%d_%d" % ff)
        else:
            ############### 1.关键决策变量 #############
            # h_i_n_k，函数i是否在机器n的核k上运行
            mdl.h = mdl.continuous_var_cube(range_func, range_server, range_core, name=lambda fsc: "h_%d_%d_%d" % fsc)
            
            # p_n_l1_l2 机器n中镜像块l1和l2的下载序列关系
            mdl.P = mdl.continuous_var_cube(range_server, range_layer, range_layer, name=lambda sll: "p_%d_%d_%d" % sll, ub=1)
            
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
            线性化辅助变量，满足如下条件，Q才为1
            1. i部署到n上
            2. n需要下载l1,l2
            3. l1下载比l2先
            '''
            Q_ = [(n,i,l1,l2) for n in range_server for i in range_func for l1 in range_layer for l2 in range_layer]
            mdl.Q = mdl.continuous_var_dict(Q_, name=lambda nill: "q_%d_%d_%d_%d" % nill, ub=1)

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
        mdl.T_image = mdl.continuous_var_list(range_func, name=lambda l: "t_image_%d" % l) # 镜像准备好时间
        mdl.TET = mdl.continuous_var()

        h,P,X,XX,B,H,T_data,T_start,T_end,T_image,G,Q,Y,TET = mdl.h,mdl.P,mdl.X,mdl.XX,mdl.B,mdl.H,mdl.T_data,mdl.T_start,mdl.T_end,mdl.T_image,mdl.G,mdl.Q,mdl.Y,mdl.TET

        M = 1e11 # 一个足够大的数

        ########## 决策 end ###########################


        ########## 约束 start ###########################

        # source和sink部署到固定的机器
        mdl.add_constraints(X[i,self.generate_pos] == 1 for i in [self.source, self.sink])
        mdl.add_constraints(h[i,self.generate_pos, 0] == 1 for i in [self.source, self.sink])

        #（1）每个函数只能部署在一个服务器上
        mdl.add_constraints(mdl.sum(X[i, n] for n in range_server) == 1 for i in range_func)

        #（4）结束时间
        mdl.add_constraints((T_end[i] == T_start[i] + \
                                mdl.sum(X[i,n] * self.func_process[i][n] for n in range_server)) \
                    for i in range_func)
        
        # （7）函数只能调度到含有所需镜像块的机器
        if image_related:
            mdl.add_constraints( X[i,n]*self.is_func_has_layer(i, l) <= G[n,l] \
                for i in range_func \
                    for n in range_server \
                        for l in range_layer)
        
        # （16） 开始时间与数据传输及镜像准备好时间的关系
        mdl.add_constraints(T_start[i] >= T_data[i] for i in range_func)
        if image_related:
            mdl.add_constraints(T_start[i] >= T_image[i] for i in range_func)

        # （17） TET
        mdl.add_constraints(TET >= T_end[i] for i in range_func)

        # (20) Q值计算
        # mdl.add_constraints( Q[n,i,l1,l2] <= (X[i,n]+P[n,l1,l2])/2\
        #                     for n in range_server \
        #                     for i in range_func \
        #                     for l1 in range_layer \
        #                     for l2 in range_layer)

        # mdl.add_constraints( Q[n,i,l1,l2] >= (X[i,n]+P[n,l1,l2]-1)/2\
        #                     for n in range_server \
        #                     for i in range_func \
        #                     for l1 in range_layer \
        #                     for l2 in range_layer)
        
        if image_related:
            mdl.add_constraints( Q[n,i,l1,l2] >= X[i,n]+P[n,l1,l2]-1 \
                                for n in range_server \
                                for i in range_func \
                                for l1 in range_layer \
                                for l2 in range_layer)
        
        
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

        if image_related:
            # (24) 和 (25)合并写
            # mdl.add_constraints( P[n,l1, l2] + P[n,l2,l1] <= (G[n, l1] + G[n, l2]) / 2  \
            #                     for n in range_server \
            #                     for l1 in range_layer \
            #                     for l2 in range_layer if l1 != l2)

            # mdl.add_constraints( P[n,l1, l2] + P[n,l2,l1] >= (G[n, l1] + G[n, l2] - 1) / 2  \
            #                     for n in range_server \
            #                     for l1 in range_layer \
            #                     for l2 in range_layer if l1 != l2)
            mdl.add_constraints( P[n,l1, l2] + P[n,l2,l1] >= G[n, l1] + G[n, l2] - 1 \
                                for n in range_server \
                                for l1 in range_layer \
                                for l2 in range_layer if l1 != l2)

            # (26)
            mdl.add_constraints( P[n,l1, l2] + P[n, l2, l3] + P[n, l3, l1] <= 2 \
                                for n in range_server \
                                for l1 in range_layer\
                                for l2 in range_layer if l2 != l1 \
                                for l3 in range_layer if l3 != l2 and l3 != l1)

            # (27)
            mdl.add_constraints( P[n,l, l] == G[n,l] \
                                for n in range_server \
                                for l in range_layer)

        # 下面三条约束计算函数i和函数j之间的带宽
        # (28)
        # mdl.add_constraints(XX[i,j,n,n_] >= (X[i,n]+X[j,n_]-1)/2\
        #     for i in range_func \
        #         for j in range_func \
        #             for n in range_server \
        #                 for n_ in range_server)
        
        # mdl.add_constraints(XX[i,j,n,n_] <= (X[i,n]+X[j,n_])/2\
        #     for i in range_func \
        #         for j in range_func \
        #             for n in range_server \
        #                 for n_ in range_server)
        mdl.add_constraints(XX[i,j,n,n_] >= X[i,n]+X[j,n_]-1\
            for i in range_func \
                for j in range_func \
                    for n in range_server \
                        for n_ in range_server)
        
        # (29) 计算 B_i_j 表示i和j之间的通信带宽
        mdl.add_constraints(B[i,j] == mdl.sum( XX[i,j,n,n_]*self.server_comm[n][n_] for n_ in range_server for n in range_server) \
                for i in range_func \
                    for j in range_func)
    
        # (30) 数据准备好时间：所有前置任务完成，并将数据发送
        mdl.add_constraints(\
            T_data[i] >= T_end[j] + B[j,i] * self.get_weight(j,i) \
                for j in range_func \
                    for i in range_func if self.G.has_edge(j,i))

        # (31) 镜像块准备好的时间：所有层都准备好
        if image_related:
            mdl.add_constraints( T_image[i] >= self.servers[n].download_latency * mdl.sum(Q[n,i,l_,l] * layers[l_].size for l_ in range_layer )\
                    for n in range_server \
                        for l in range_layer \
                            for i in range_func if self.is_func_has_layer(i, l) == 1)
        
        # 目标
        mdl.minimize(TET)
        mdl.print_information()

        solution = mdl.solve()

        return mdl, solution


    def get_func_layers(self, func_ids):
        s = set()
        for i in func_ids:
            s = s.union(set(self.funcs[i].layer))
        return list(s)

    def sequence_strategy(self, place):
        pid = 0
        for i in range(self.K):
            funcs_idx = []
            h = {}
            for _ in range(self.servers[i].core):
                funcs_idx.extend(place[pid])
                for f in place[pid]:
                    h[f] = pid
                pid+=1 # 切换到下一个Processor
            print(h, funcs_idx, self.get_func_layers(funcs_idx))
            layers_idx = self.get_func_layers(funcs_idx)
            self.solve_sequence_problem(h, i, funcs_idx, layers_idx)
        exit(0)
    
    '''
    @Params
        h: h[i] 表示函数i的部署的核
        n: 在哪台机器部署
        funcs_idx: 部署到机器n的所有函数下标，如[0, 3, 4, 7]
        layers_idx: 需要下载的layer, 如[1 ,3, 5, 7]，表示需要下载1,3,5,7
    @Return
        download sequence, 如
    '''
    def solve_sequence_problem(self, h, n, funcs_idx, layers_idx):
        mdl = Model(name="propose_solver")
        range_func = range(len(funcs_idx))
        range_layer = range(len(layers_idx))

        ### 初始记映射关系
        funcs_map = {} # func_id -> idx
        for idx, fid in enumerate(funcs_idx):
            funcs_map[fid] = idx
        layers_map = {} # layer_id -> idx
        for idx, lid in enumerate(layers_idx):
            layers_map[lid] = idx

        def get_func_id(idx): return funcs_idx[idx]
        def get_layer_id(idx): return layers_idx[idx]

        # p_l1_l2 机器n中镜像块l1和l2的下载序列关系
        mdl.P = mdl.continuous_var_matrix(range_layer, range_layer, name=lambda ll: "p_%d_%d" % ll, ub=1)

        mdl.T_start = mdl.continuous_var_list(range_func, name=lambda l: "t_start_%d" % l) # 开始时间
        mdl.T_end = mdl.continuous_var_list(range_func, name=lambda l: "t_end_%d" % l) # 结束时间
        mdl.T_image = mdl.continuous_var_list(range_func, name=lambda l: "t_image_%d" % l) # 镜像准备好时间
        mdl.TET = mdl.continuous_var()

        P,T_start,T_end,T_image,TET = mdl.P, mdl.T_start, mdl.T_end, mdl.T_image, mdl.TET 
        
        # (1) T_end
        mdl.add_constraints((T_end[i] == T_start[i] + self.func_process[get_func_id(i)][n]) for i in range_func)

        # (2) 任务之间的序列关系
        for i in range_func:
            for func_j in self.G.predecessors(get_func_id(i)):
                if(func_j in funcs_map):
                    mdl.add_constraint(T_start[i] >= T_end[funcs_map[func_j]])

        # (3) 
        mdl.add_constraints(T_start[i] >= T_image[i] for i in range_func)

        # (4)
        mdl.add_constraints(TET >= T_end[i] for i in range_func)

        # (5) T_image[i] -> 函数I的image准备好时间
        for i in range_func:
            # 获取当前func所需的所有layer
            for layer_l in self.funcs[get_func_id(i)].layer:
                mdl.add_constraint(\
                T_image[i] >= self.servers[n].download_latency \
                    * mdl.sum(P[l_,layers_map[layer_l]] \
                            * self.layers[get_layer_id(l_)].size \
                        for l_ in range_layer ))
        
        # (6)
        for i in range_func:
            for j in range_func:
                if(i!=j and h[get_func_id(i)] == h[get_func_id(j)]):
                    mdl.add_constraint(mdl.max(T_start[i]-T_end[j], T_start[j] - T_end[i]) >= 0)

        for l1 in range_layer:
            for l2 in range_layer:
                if(l1 == l2):
                    # (9)
                    mdl.add_constraint(P[l1,l1] == 1)
                else:
                    # (7)
                    mdl.add_constraint(P[l1,l2] + P[l2, l1] == 1)
        
        for l1 in range_layer:
            for l2 in range_layer:
                for l3 in range_layer:
                    if(l1!=l2!=l3):
                        # (8)
                        mdl.add_constraint(P[l1,l2] + P[l2, l3] + P[l3, l1] <= 2)

        mdl.minimize(TET)
        mdl.print_information()

        solution = mdl.solve()

        print(solution)

        # P = [[mdl.P[l1,l2].solution_value for l2 in range_layer] for l1 in range_layer]
        # self.logger.debug(f"P[{n}] = ")
        # self.logger.debug(np.array(P))
        exit(0)



    def __init__(self, data, config):
        super().__init__(data, config)

    def schedule(self):
        place = super().schedule()[1]
        self.sequence_strategy(place)
        exit(0)
        # mdl, solution = self.solve()
        # self.parse(mdl, solution)
        # return self.output_scheduler_strategy()
