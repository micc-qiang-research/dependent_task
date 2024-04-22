import networkx as nx
import pseudoflow
import numpy as np
# import logging

# level = logging.INFO
# logging.basicConfig(level=level, format='%(levelname)s - %(message)s')

class SidneyDecomposition:
    def __init__(self, n, dag, W, P):
        self.n = n
        self.dag = dag
        self.W = W
        self.P = P

    def node_id(self,i):
            return i + 1
    def node_id_rev(self, n):
        return n-1

    def build_graph(self):
        inf = 10**10 
        rg_min = 0.
        self.P = np.maximum(np.array(self.P), 1e-10) # 防止P出现0
        rg_max = np.max(np.array(self.W)/(np.array(self.P))) + 0.01
        # rg_max = sys.maxsize
        source = 0
        sink = self.n + 1

        G = nx.DiGraph()

        # extern
        for i in range(self.n):
            G.add_edge(source, self.node_id(i), capacity=-self.W[i], mult=self.P[i])
            G.add_edge(self.node_id(i),sink,  capacity=self.W[i], mult=-self.P[i])

        # inner
        if type(self.dag).__name__ == 'set': # {(1,2), (2,3), ...}
            for i,j in self.dag:
                    G.add_edge(self.node_id(i), self.node_id(j), capacity=inf, mult=1)
        else:
            for i in range(self.n):
                for j in range(self.n):
                    if self.dag[i][j] == 1:
                        G.add_edge(self.node_id(i), self.node_id(j), capacity=inf, mult=1)
        
        lambda_range = [rg_min, rg_max]

        return source, sink, lambda_range, G

    def run(self):
        source, sink, lambda_range, G = self.build_graph()
        # print(source, sink, lambda_range, G)

        breakpoints, cuts, info = pseudoflow.hpf(
            G,  # Networkx or igraph directed graph.
            source,  # Node id of the source node.
            sink,  # Node id of the sink node.
            const_cap="capacity",  # Edge attribute with the constant capacity.
            mult_cap="mult",  # Edge attribute with the lambda multiplier.
            lambdaRange=lambda_range,  # (lower, upper) bounds for the lambda parameter.
            roundNegativeCapacity=True  # True if negative arc capacities should be rounded to zero.
        )

        # logging.debug("breakpoints: {}".format(breakpoints)) 
        # logging.debug("cuts: {}".format(cuts)) 

        #### 分析结果
        idx = {i: np.min(np.nonzero(cuts[i]))  for i in cuts.keys() if i != source and i != sink}
        # idx = {i: 1  for i in cuts.keys() if i != source and i != sink}

        res = {}
        for i in idx.values():
            res[i] = []
            for j in idx.keys():
                if idx[j] == i:
                    res[i].append(self.node_id_rev(j))

        Y = []
        for i in sorted(res.keys(), reverse=True):
            Y.append(res[i])        
        # logging.debug("Y: {}".format(Y))
        return Y

def acquire_data1():
    n = 4
    dag = [[0 for i in range(n)] for j in range(n)]
    dag[0][1] = 1
    dag[0][2] = 1
    dag[1][3] = 1
    W = [1,2,2,3]
    P = [2,2,3,1]
    return n, np.array(dag), np.array(W), np.array(P)


def acquire_data2():
    n = 4
    dag = [[0 for i in range(n)] for j in range(n)]
    dag[0][1] = 1
    dag[1][2] = 1
    dag[2][3] = 1
    W = [4,3,2,1]
    P = [2,2,2,2]
    return n, np.array(dag), np.array(W), np.array(P)

def acquire_data3():
    n = 4
    dag = set()
    dag.add((0,1))
    dag.add((1,2))
    dag.add((2,3))
    W = [4,3,2,1]
    P = [2,2,0,2]
    return n, dag, np.array(W), np.array(P)


###############  data end #########

if __name__ == "__main__":
    acquire_data = acquire_data2
    SidneyDecomposition(*acquire_data()).run()