from abc import abstractmethod, ABCMeta
import networkx as nx
import numpy as np
import json
from typing import NamedTuple

Server = NamedTuple("Server", [("core", int), ("storage", float), ("download_latency", float)])
Func = NamedTuple("Func", [("layer", set)])
Layer = NamedTuple("Layer", [("size", float)])

class Data(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, path):
        pass
        # self.N = None # task number, include source、sink
        # self.K = None # server number， include cloud
        # self.L = None # layer number

        ## 如func_info为 [[0,1],[1,2,3],[0,2]]，表示第0个函数包含第0、1层，第1个函数包含第1、2、3层，第2个函数包含第0、2层
        # self.G = None # DAG
        # self.func_info = None # 每个函数包含哪些layer
        # self.func_process = None # 每个节点的处理时间
        # self.func_prepare = None # 每个节点的环境准备时间，本项目为0，忽略
        
        # self.server_comm = None # server之间通信带宽
        # self.server_info = None # 每台server含有的core数量、存储大小、下载延迟
        
        # self.layer_info = None # 每个layer的大小

        # self.generate_pos = None # dag生成位置


        ### 衍生变量
        # self.func_startup = None # 每个节点的环境大小
        # self.servers
        # self.layers
        # self.funcs

    @classmethod
    def check(cls, dump=False):
        def check_param(method):
            def wrapper(*args, **kwargs):
                method(*args, **kwargs)  
                self = args[0]
                if not isinstance(self, cls):
                    raise BaseException("method is not a Data instance")
                assert hasattr(self, "N") and \
                        hasattr(self,"K") and \
                        hasattr(self,"L") and \
                        hasattr(self,"G") and \
                        hasattr(self,"func_info") and \
                        hasattr(self,"func_process") and \
                        hasattr(self,"server_comm") and \
                        hasattr(self,"server_info") and \
                        hasattr(self,"layer_info") and \
                        hasattr(self,"generate_pos"), "error"
                # self.func_startup
                # self.servers
                # self.layers
                # self.funcs
                
                assert type(self.N) == int and type(self.K) == int and type(self.L) == int, "error"
                assert len(self.func_info) == self.N
                assert np.array(self.func_process).shape == (self.N, self.K), "error"
                assert np.array(self.server_comm).shape ==  (self.K, self.K), "error"
                assert np.array(self.server_info).shape == (self.K, 3), "error"
                assert len(self.layer_info) == self.L, "error"
                assert 0 <= self.generate_pos < self.K, "error"

                self.funcs = [Func(f) for f in self.func_info]
                self.servers = [Server(*s) for s in self.server_info]
                self.layers = [Layer(l) for l in self.layer_info]
                self.func_startup = []
                for i in range(self.N):
                    size = 0
                    for l in range(self.L):
                        if l in self.funcs[i].layer:
                            size += self.layers[l].size
                    self.func_startup.append(size)

                if dump:
                    self.dump()
            return wrapper
        return check_param

    def dump(self):
        info = {
            "_N": self.N,
            "_K": self.K,
            "_L": self.L,
            "_G": list(self.get_edges()),
            "_func_info": [list(i) for i in self.func_info],
            "_func_process": list(self.func_process),
            "_server_comm": list(self.server_comm),
            "_server_info": list(self.server_info),
            "_layer_info": list(self.layer_info),
        }
        separators = (',', ':')
        with open("data.json", "w") as f:
            json.dump(info, f, indent=2, separators=separators)

    def get_edges(self):
        edges = []
        for u,v,data in self.G.edges(data=True):
            edges.append([u,v,data["weight"]])
        return edges
