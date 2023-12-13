from typing import List, Dict
import pandas as pd
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
from .data import Data
import json


class DataByCsv(Data):
    @Data.check(dump=True)
    def __init__(self, path: str):
        self.init_default_cluster_info()
        self.csv_path = path
        self.df = pd.read_csv(self.csv_path, dtype={'dm': str})

        # step: node id, index, virtual root 相关
        self.node_id_tuple = tuple([str(i) for i in sorted(set(self.df['dm']))])
        self.root_nid = self.df.at[0, 'dm']
        self.node_num = len(self.node_id_tuple)     # 包含 "-1"
        self.N = len(self.node_id_tuple)+1 # include sink
        self.sink = self.N - 1

        # step: edge id, cpu, route cost, cpu
        edge_id_list = list(sorted(list(self.df['edge_id'])))
        edge_id_list.remove("-1.-1")
        self.edge_id_tuple = tuple(edge_id_list)

        # step: node间连接关系
        self.node_parent_dict, self.node_children_dict = self.init_parent_and_children()

        self.edge_cpu_dict: Dict[str, int] = {}
        self.edge_route_cost_dict: Dict[str, int] = {}
        for index, row in self.df.iterrows():
            edge_id = row['edge_id']
            cpu = row['cpu']
            route_cost = row["route_cost"]
            self.edge_cpu_dict[edge_id] = cpu
            self.edge_route_cost_dict[edge_id] = route_cost

        self.init_func_startup() # func_startup
        self.init_dag() # G
        self.init_func_process()

    def init_default_cluster_info(self):
        self.K = 4
        self.edge_bandwidth = [2.81, 2.86, 1.07]
        self.server_comm = [
            [0, 0.46, 0.23, 3.3],
            [0.46,0,0.24,3.3],
            [0.23,0.24,0,3.3],
            [3.3,3.3,3.3,0]
        ]
        # TODO. 每台server的核数先写死为2
        self.cores = []
        for i in range(self.K-1):
            self.cores.append(2)
        
        self.uc_comm = np.array(self.server_comm[self.K-1][:-1]).mean()
        
        self.ue_comm = [np.sum(i[:-1]) / (self.K-1) for i in self.server_comm[:-1]]

    def init_func_startup(self):
        self.image_path = os.path.join(os.path.dirname(self.csv_path), "../microservice_json")
        # step: node ms_name, cpu, route_cost相关
        self.node_ms_dict = {}
        self.func_startup = [0] * self.N
        for index, row in self.df.iterrows():
            node_id = row['dm']
            ms_name = row['ms_name']
            self.node_ms_dict[node_id] = ms_name
            if node_id != self.root_nid:
                self.func_startup[int(node_id)+1] = self.get_image_download_size(ms_name) / 1e8

    def init_func_process(self):
        self.func_process = np.zeros((self.N, self.K))
        for i in range(1,self.N-2):
            self.func_process[i] = self.get_normal_distributed_data(5,2,self.K)
        self.func_process[self.N-2] = self.get_normal_distributed_data(3,2,self.K)
        self.func_prepare = 0.2 * np.sum(np.array(self.func_process), axis=1) / np.array(self.func_process).shape[1]

    def init_dag(self):
        # step: 初始化G
        self.G = nx.DiGraph()
        for _, row in self.df.iterrows():
            edge_id = row['edge_id']

            # 解析 edge_id，获取 parent_nid 和 nid
            parent_nid, nid = edge_id.split('.')
            if parent_nid == nid:
                continue

            # 添加边
            self.G.add_edge(int(parent_nid)+1, int(nid)+1, weight=self.edge_route_cost_dict[edge_id])

        for nid in self.node_id_tuple:
            if len(self.node_children_dict[nid]) == 0:
                self.G.add_edge(int(nid)+1, self.N-1, weight=0)

        # self.draw_df()


    def get_image_download_size(self, file):
        if not hasattr(self, "ms_imagesize_dict"):
            self.ms_imagesize_dict = {}
        if file in self.ms_imagesize_dict:
            return self.ms_imagesize_dict[file]
        with open(os.path.join(self.image_path, file), 'r') as f:
            layers = json.load(f)
            doownload_size = np.array(list(layers.values()), dtype="float").sum()
            self.ms_imagesize_dict[file] = doownload_size
        return self.ms_imagesize_dict[file]

    def get_normal_distributed_data(self, mu, sigma, size):
        data = np.random.normal(mu, sigma, size)
        return np.maximum(data, 1)


    def get_msname(self, nid):
        return self.node_ms_dict[nid]


    def get_edge_cpu(self, parent_nid, nid):
        return self.edge_cpu_dict[f"{parent_nid}.{nid}"]

    def get_parent(self, nid):
        return tuple(self.node_parent_dict[nid])   # root_nid -> [virtual_root]

    def get_children(self, nid):
        return tuple(self.node_children_dict[nid])

    def get_ms_list(self):
        result = set()
        for nid in self.node_id_tuple:
            result.add(self.node_ms_dict[nid])
        return list(result)

    def get_edge_route_cost(self, parent_nid, nid):
        return self.edge_route_cost_dict[f"{parent_nid}.{nid}"]
        # parent_id = self.get_parent(nid)
        # return self.node_route_array[self.node_index_dict[parent_id], self.node_index_dict[nid]]

    # tag: 展示信息
    def draw_df(self):
        # 使用graphviz_layout定义树形图的布局
        pos = nx.drawing.nx_agraph.graphviz_layout(self.G, prog='dot')
        # pos = nx.nx_agraph.graphviz_layout(self.G)

        # 绘制树形图
        nx.draw_networkx(self.G, pos, with_labels=True, node_size=500, node_color='lightblue', arrows=True)
        weight = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=weight)
        plt.show()

        return True

    def get_route_array(self):
        return self.node_route_array.copy()

    def init_parent_and_children(self):
        parent_dict = {
            nid: [] for nid in self.node_id_tuple
        }
        children_dict = {
            nid: [] for nid in self.node_id_tuple
        }

        for edge_id in self.edge_id_tuple:
            parent_id, child_id = edge_id.split('.')
            parent_id = parent_id
            child_id = child_id
            if parent_id != child_id:
                parent_dict[child_id].append(parent_id)
                children_dict[parent_id].append(child_id)

        return parent_dict, children_dict

if __name__ == "__main__":
    app = Application("app_1.csv", "./application_csv")
    app.draw_df()
    # app.save_fig("./test.png")