import networkx as nx
import pandas as pd
import numpy as np
import os

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

        # 准备时间设置成平均执行时间的1/5
        self.func_prepare = 0.2 * np.sum(np.array(self.func_process), axis=1) / np.array(self.func_process).shape[1]

        self.server_comm = []
        for i in range(self.K):
            comm = []
            for j in range(self.K):
                comm.append(data.get(float))
            self.server_comm.append(comm)

        # 用户和云的通信延迟为：云和其他边缘服务器通信延迟的均值
        self.uc_comm = np.array(self.server_comm[self.K-1][:-1]).mean()
        
        # 用户和第i个边缘服务器之间的通信延迟为：第i个边缘服务器和其他边缘服务器通信延迟的均值
        self.ue_comm = [np.sum(i[:-1]) / (self.K-1) for i in self.server_comm[:-1]]
        
        return os.path.join(os.path.dirname(path),data.get(str))


    def acquire_data(self, path):
        dag = self.read_data(path)
        self.read_dag(dag)
