from .data import Data
import json
import networkx as nx


class DataByJson(Data):
    @Data.check()
    def __init__(self, path):
        with open(path, "r") as f:
            self.__dict__ = json.load(f)
            edges = self.edge_list
            self.G = nx.DiGraph()
            self.G.add_weighted_edges_from(edges)

