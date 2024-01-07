import networkx as nx
import pandas as pd
import numpy as np
import os
from .data import Data

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


class DataByTxt(Data):
    @Data.check(True)
    def __init__(self, path):
        self.read_data(path)

    def read_data(self, path):
        data = DataSource(path)
        self.N = data.get(int) # func number
        self.K = data.get(int) # server number，include cloud
        self.L = data.get(int) # layer number

        self.func_info = []
        for i in range(self.N):
            # layer_info + func_prepare
            self.func_info.append([int(i) for i in data.get(str).split(",")])

        self.G = nx.DiGraph()
        edges_number = data.get(int)
        for i in range(edges_number):
            s,d,w = data.get(int), data.get(int), data.get(float)
            self.G.add_edge(s,d,weight=w)
        
        self.func_process = []
        for i in range(self.N):
            tmp = []
            for j in range(self.K):
                tmp.append(data.get(float))
            self.func_process.append(tmp)

        self.server_comm = []
        for i in range(self.K):
            comm = []
            for j in range(self.K):
                comm.append(data.get(float))
            self.server_comm.append(comm)

        self.server_info = []
        for i in range(self.K):
            # core、storage、download_latency
            tmp = []
            tmp.append(data.get(int))
            tmp.append(data.get(float))
            tmp.append(data.get(float))
            self.server_info.append(tmp)
        
        self.layer_info = []
        for i in range(self.L):
            self.layer_info.append(data.get(float))

        self.generate_pos = data.get(int) # dag生成位置

