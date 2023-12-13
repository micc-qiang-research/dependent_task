from abc import abstractmethod, ABCMeta
import networkx as nx
import numpy as np
import json

class Data(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, path):
        pass
        # self.N = None # 任务数，包括source sink
        # self.K = None # edge server数
        # self.G = None # DAG
        # self.func_process = None # 每个节点的处理时间
        # self.func_prepare = None # 每个节点的环境准备时间
        # self.func_startup = None # 每个节点的环境大小
        # self.edge_bindwidth = None # 每个节点下载image带宽
        # self.server_comm = None # server之间通信带宽
        # self.ue_comm = None
        # self.uc_comm = None
        # self.cores = None # 每个edge server含有的core数量

    @classmethod
    def check(cls, dump=False):
        def check_param(method):
            def wrapper(*args, **kwargs):
                method(*args, **kwargs)  
                self = args[0]
                if not isinstance(self, cls):
                    raise BaseException("method is not a Data instance")
                self.N
                self.K
                self.G
                self.func_process
                self.func_prepare
                self.func_startup
                self.edge_bandwidth
                self.server_comm
                self.ue_comm
                self.uc_comm
                self.cores
                assert type(self.N) == int, "error"
                assert type(self.K) == int, "error"
                assert np.array(self.func_process).shape == (self.N, self.K), "error"
                assert len(self.func_prepare) == self.N, "error"
                assert len(self.func_startup) == self.N, "error"
                assert len(self.edge_bandwidth) == self.K - 1, "error"
                assert np.array(self.server_comm).shape ==  (self.K, self.K), "error"
                assert len(self.ue_comm) == self.K - 1, "error"
                assert len(self.cores) == self.K - 1, "error"
                if dump:
                    self.dump()
            return wrapper
        return check_param

    def dump(self):
        info = {
            "_N": self.N,
            "_K": self.K,
            "_G": list(self.get_edges()),
            "_func_process": [list(i) for i in self.func_process],
            "_func_prepare": list(self.func_prepare),
            "_func_startup": list(self.func_startup),
            "_edge_bandwidth": list(self.edge_bandwidth),
            "_server_comm": list(self.server_comm),
            "_ue_comm": list(self.ue_comm),
            "_uc_comm": self.uc_comm,
            "_cores": list(self.cores),
        }
        separators = (',', ':')
        with open("data.json", "w") as f:
            json.dump(info, f, indent=2, separators=separators)

    def get_edges(self):
        edges = []
        for u,v,data in self.G.edges(data=True):
            edges.append([u,v,data["weight"]])
        return edges

    @property
    def N(self) -> int:
        if not hasattr(self, "_N"):
            raise Exception("N is not defined")
        return self._N

    @N.setter
    def N(self, value: list):
        self._N = value

    @property
    def K(self) -> list:
        if not hasattr(self, "_K"):
            raise Exception("K is not defined")
        return self._K

    @K.setter
    def K(self, value: int):
        self._K = value

    @property
    def G(self) -> nx.DiGraph:
        if not hasattr(self, "_G"):
            raise Exception("G is not defined")
        return self._G

    @G.setter
    def G(self, value: nx.DiGraph):
        self._G = value

    @property
    def func_process(self) -> list:
        if not hasattr(self, "_func_process"):
            raise Exception("func_process is not defined")
        return self._func_process

    @func_process.setter
    def func_process(self, value: list):
        self._func_process = value

    @property
    def func_prepare(self) -> list:
        if not hasattr(self, "_func_prepare"):
            raise Exception("func_prepare is not defined")
        return self._func_prepare

    @func_prepare.setter
    def func_prepare(self, value: list):
        self._func_prepare = value

    @property
    def func_startup(self) -> list:
        if not hasattr(self,"_func_startup"):
            raise Exception("func_startup is not defined")
        return self._func_startup

    @func_startup.setter
    def func_startup(self, value: list):
        self._func_startup = value

    @property
    def edge_bandwidth(self) -> list:
        if not hasattr(self,"_edge_bandwidth"):
            raise Exception("edge_bandwidth is not defined")
        return self._edge_bandwidth

    @edge_bandwidth.setter
    def edge_bandwidth(self, value: list):
        self._edge_bandwidth = value

    @property
    def server_comm(self) -> list:
        if not hasattr(self, "_server_comm"):
            raise Exception("server_comm is not defined")
        return self._server_comm

    @server_comm.setter
    def server_comm(self, value: list):
        self._server_comm = value

    @property
    def ue_comm(self) -> int:
        if not hasattr(self, "_ue_comm"):
            raise Exception("ue_comm is not defined")
        return self._ue_comm

    @ue_comm.setter
    def ue_comm(self, value: int):
        self._ue_comm = value

    @property
    def uc_comm(self) -> int:
        if not hasattr(self, "_uc_comm"):
            raise Exception("uc_comm is not defined")
        return self._uc_comm

    @uc_comm.setter
    def uc_comm(self, value: int):
        self._uc_comm = value

    @property
    def cores(self) -> list:
        if not hasattr(self, "_cores"):
            raise Exception("cores is not defined")
        return self._cores

    @cores.setter
    def cores(self, value: list):
        self._cores = value
