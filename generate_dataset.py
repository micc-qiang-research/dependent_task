import numpy as np
import json
from glob import glob
from os import path
from collections.abc import Iterable
from tqdm import tqdm



_t = 1 # 平均执行时间， cloud: 0.75t
_b = 1 # 下载镜像的延迟 cloud: 0.5b
_c = range(1, 4) # 核数量 cloud:4
_e = 0.5 # 默认edge权重
_l = 1 # layer大小，归一化为1
# _d = 1 # server之间传送数据的开销 _e*_d / _t = CCR

import pydot
import numpy as np
from random import randint, gauss

def round_2(f):
    def wrapper(*args, **kwargs):
        res = f(*args, **kwargs)
        if isinstance(res, Iterable):
            return [round(r, 2) for r in res]
        return round(res, 2)
    return wrapper

@round_2
def get_computation_time(size=1):
    return np.random.uniform(0.5*_t, 1.5*_t, size=size).tolist()

@round_2
def get_node_weight():
    return np.random.uniform(0.5*_e, 1.5*_e)

@round_2
def get_download_latency():
    return np.random.uniform(0.5*_b, 1.5*_b)

@round_2
def get_commucation_time(ccr=0.5):
    _d = ccr * _t / _e
    return np.random.uniform(0.5*_d, 1.5*_d)

@round_2
def get_layer_size(size=1):
    return np.random.uniform(0.5*_l, 1.5*_l, size=size).tolist()

# 随机生成函数依赖的镜像块信息
@round_2
def get_layer_info(L=10, dcr=5):
    _size = dcr * _t / _l # _l * _size / _t = dcr
    size = max(min(int(np.random.normal(_size, _size/3)),L),1)
    res = np.random.choice(range(L), size=size, replace=False) # replace=False表示不重复采样
    res.sort()
    return res.tolist()

'''
Input:
    K: server的数量
    L: layer的数量
    ccr: commucation to computation ratio
    dcr: download to computation ratio
    lfr: layer number to function number ratio
'''
def build_data(filename, K=3, lfr=2, ccr=0.5, dcr=5):
    graph = pydot.graph_from_dot_file(filename)[0]
    n_nodes = len(graph.get_nodes())

    # get adjacency matrix for DAG
    adj_matrix = np.full((n_nodes, n_nodes), -1)
    n_edges = 0
    for e in graph.get_edge_list():
        adj_matrix[int(e.get_source())-1][int(e.get_destination())-1] = 0
        n_edges += 1

    # if DAG has multiple entry/exit nodes, create dummy nodes in its place
    ends = np.nonzero(np.all(adj_matrix==-1, axis=1))[0]    # exit nodes
    starts = np.nonzero(np.all(adj_matrix==-1, axis=0))[0]  # entry nodes
    start_node = pydot.Node("0", alpha="\"0\"", size="\"0\"")
    end_node = pydot.Node(str(n_nodes+1), alpha="\"0\"", size="\"0\"")
    graph.add_node(start_node)
    graph.add_node(end_node)

    for start in starts:
        s_edge = pydot.Edge("0", str(start+1), size="\"0\"")
        graph.add_edge(s_edge)
        
    for end in ends:
        e_edge = pydot.Edge(str(end+1), str(n_nodes+1), size="\"0\"")
        graph.add_edge(e_edge)

    n_nodes = len(graph.get_nodes())
    L = int(lfr*n_nodes)

    # construct computation matrix
    # comp_matrix = np.empty((n_nodes, K))
    comp_matrix = [[0 for j in range(K)]for i in range(n_nodes)]
    for n in graph.get_node_list():
        comp_temp = get_computation_time(size=K)
        comp_matrix[int(n.get_name())][:] = comp_temp

    #get modified adjency matrix
    edge_list = []
    for e in graph.get_edge_list():
        edge_list.append([int(e.get_source()), \
                           int(e.get_destination()), 
                           get_node_weight()])

    # for n in graph.get_node_list():
    #     print(n.get_name())
    func_info = []
    for i in range(n_nodes):
        if i == 0 or i == n_nodes-1:
            func_info.append([]) # source和sink任务没有启动延迟，不依赖镜像块
        else:
            func_info.append(get_layer_info(L=L, dcr=dcr))
        # print(layer_info)

    layer_info = get_layer_size(size=L)

    generate_pos = 0

    server_comm = [[0 for j in range(K)] for i in range(K)]
    for i in range(K):
        for j in range(i+1,K):
            server_comm[i][j] = server_comm[j][i] = get_commucation_time(ccr=ccr)

    server_info = [[] for i in range(K)]
    for i in range(K):
        server_info[i].append(int(np.random.choice(_c))) # 核数量
        server_info[i].append(1e8) # 假设storage是无穷的
        server_info[i].append(get_download_latency()) # 下载镜像延迟

    # 修正cloud参数
    _d = ccr * _t / _e
    for i in range(K-1):
        server_comm[i][K-1] = server_comm[K-1][i] = 15*_d
    for i in range(n_nodes):
        comp_matrix[i][K-1] = 0.75 * _t
    server_info[K-1][0] = 4

    
    return {"N": n_nodes,      # 函数个数
            "K": K,            # server个数
            "L": L,            # 镜像块个数
            "func_info": func_info,    # 函数信息：所需Layer
            "edge_list": edge_list,   # DAG的邻接矩阵
            "func_process": comp_matrix,  # 函数在各个server上的处理时间
            "server_comm": server_comm,  # server之间通信延迟
            "server_info": server_info,  # server信息：核数、storage大小、下载镜像带宽
            "layer_info": layer_info,   # layer大小信息
            "generate_pos": generate_pos  # 生成位置
    }


if __name__ == "__main__":
    # data = build_data('dag/10_0.1_0.2_0.2_1.dot')
    # separators = (',', ':')
    # with open("data.json", "w") as f:
    #     json.dump(data, f, indent=2, separators=separators)

    output_dir = "./data/json/"
    
    filenames = glob('dag/*.dot')
    
    for filename in tqdm(filenames):
        data = build_data(filename)
        separators = (',', ':')
        with open(path.join(output_dir, path.basename(filename)[:-3])+"json", "w") as f:
            json.dump(data, f, indent=2, separators=separators)
        