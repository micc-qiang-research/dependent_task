import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from queue import PriorityQueue

# 读取txt，每次返回一个值
class TxtReader:
    def __init__(self, path):
        self.array = []
        with open(path, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                self.array.extend(line.split())
        self.idx = 0
        self.length = len(self.array)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.array[idx]

    def __iter__(self):
        return iter(self.array)

class DataSource:
    def __init__(self, path):
        self.data = iter(TxtReader(path))

    def get(self,type=int):
        if type == int:
            return int(next(self.data))
        elif type == float:
            return float(next(self.data))
        elif type == str:
            return str(next(self.data))
        else:
            raise "type error"


class Data:
    def __init__(self, path):
        self.path = path
        self.acquire_data(path)

    def read_dag(self, path):
        self.G = nx.DiGraph()
        data = pd.read_csv(path)
        edges = [(int(s)-1,int(d)-1,{"weight": w}) for s,d,w in data.to_numpy()]
        self.G.add_weighted_edges_from(edges)


    def read_data(self, path):
        data = DataSource(path)
        self.K = data.get(int) # server number，include cloud
        self.N = data.get(int) # func number, not include source\sink
        
        self.edge_bandwidth = []
        for i in range(self.K-1):
            self.edge_bandwidth.append(data.get(float))
        
        self.func_startup = []
        for i in range(self.N):
            self.func_startup.append(data.get(float))
        
        self.func_process = []
        for i in range(self.N):
            process = []
            for j in range(self.K):
                process.append(data.get(float))
            self.func_process.append(process)

        self.server_comm = []
        for i in range(self.K):
            comm = []
            for j in range(self.K):
                comm.append(data.get(float))
            self.server_comm.append(comm)
        
        return os.path.join(os.path.dirname(path),data.get(str))


    def acquire_data(self, path):
        dag = self.read_data(path)
        self.read_dag(dag)

    def draw_dag(self):
        G = self.G
        pos = nx.nx_agraph.graphviz_layout(G)
        weights = nx.get_edge_attributes(G, "weight")
        weights = {e: weights[e]["weight"] for e in weights}
        nx.draw_networkx(G, pos, with_labels=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
        plt.show()


class PQueue:
    def __init__(self):
        self.queue = PriorityQueue()
        self.label = set()

    def get(self):
        return self.queue.get()        

    def put(self, item):
        if item[1] in self.label:
            return    
        self.label.add(item[1])
        return self.queue.put(item)

    def empty(self):
        return self.queue.empty()


class SDTS:
    def __init__(self, data):
        self.data = data

    # 根据函数环境大小即边缘带宽，计算下载时间
    def get_func_edge_download(self, func_startup, edge_bandwith):
        return (np.tile(func_startup.reshape(len(func_startup), 1), (1, self.data.K-1)).T / edge_bandwith.reshape(-1,1)).T
    
    # 得到边的权重
    def get_weight(self, G, s, d):
        res = G.edges[s,d]["weight"]["weight"]
        return res

    def priority(self, G, func_edge_download, edge_bandwith, func_process):
        vertices = list(nx.topological_sort(G))
        vertices.reverse() # 计算优先级的次序
        N = self.data.N
        priority_dict = { i: 0 for i in range(N)}
        assert N == len(vertices), "N != len(vertices)"

        d_mean = sum(edge_bandwith) / (self.data.K - 1)**2 # edge server之间的平均带宽

        for v in vertices:
            if v == N - 1 or v == 0:
                continue
            t_mean = func_process[v].mean() # func v的平均处理时间
            t_d_mean = func_edge_download[v].mean() # func v的平均下载时间
            for s in G.successors(v):
                priority_dict[v] = max(priority_dict[v], self.get_weight(G, v, s) * d_mean + priority_dict[s])
            priority_dict[v] = priority_dict[v] + t_mean + t_d_mean
        return priority_dict
    

    def sdts(self):
        G = self.data.G
        func_process = np.array(self.data.func_process)
        edge_bandwith = np.array(self.data.edge_bandwidth) # 1..(k-1)
        func_startup = np.array(self.data.func_startup)
        func_edge_download = self.get_func_edge_download(func_startup, edge_bandwith)

        t_k_c = [0] * (self.data.K - 1) # edge server当前下载完成时间
        priority_dict =  self.priority(G, func_edge_download, edge_bandwith, func_process)

        L = PQueue()
        L.put((-priority_dict[0],0))
        N = self.data.N
        h = [-1] * N
        
        while not L.empty():
            _, v = L.get()
            if v == 0 or v == N - 1:
                pass
            else:
                pass

            for s in G.successors(v):
                L.put((-priority_dict[s],s))
            print(v)


if __name__ == '__main__':
    data = Data("./data/data_1.txt")
    # data.draw_dag()
    # print(data.func_process)
    SDTS(data).sdts()