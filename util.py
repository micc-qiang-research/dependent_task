
import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue

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

def draw_dag(G):
    pos = nx.nx_agraph.graphviz_layout(G)
    weights = nx.get_edge_attributes(G, "weight")
    weights = {e: weights[e]["weight"] for e in weights}
    nx.draw_networkx(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.show()