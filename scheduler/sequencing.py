
import networkx as nx
import numpy as np
from abc import abstractmethod, ABCMeta
from sequence.Lcaa import Lcaa

class GenStrategy:
    DUMB = 0
    TOPOLOGY = 1
    CUSTOM = 2

    def __init__(self, executor):
        self.G = executor.G
        self.source = executor.source
        self.sink = executor.sink


    def get_gen_strategy(self, strategy):
        match(strategy):
            case GenStrategy.DUMB:
                return self.dumb_gen_strategy
            case GenStrategy.TOPOLOGY:
                return self.topology_gen_strategy
            case GenStrategy.CUSTOM:
                return self.custom_gen_strategy
            case _:
                assert False, "gen_strategy error"

    '''
    将调度算法策略转换为本项目策略
    通用的算法策略用一个二维数组表示
    strategy[K] = [
          [0,1,5],
          [2,4,3],
           ...
    ]
    其中strategy[k][i]表示第i个core的第j个任务
    core的编号顺序为: 
        server0.core0  server0.core1 ...
        server1.core0  server1.core1 ...
        ...
        cloud
    返回:
        (func, procs)元组， procs是一个list,因为一个函数可部署到多个位置
    '''

    # 按核顺序来返回函数
    # 对于SDTS，可能前置任务没全部做完后置任务就开始了
    def dumb_gen_strategy(self, raw_strategy, order=None):
        assert len(raw_strategy) >= self.cluster.get_total_core_number(), "strategy length don't match core number"
        finished_func = set()
        all_func = set(self.G.nodes())
        pos = [0 for i in range(len(raw_strategy))]
        def is_exist(func_id):
            for i,s in enumerate(raw_strategy):
                if pos[i] >= len(s):
                    continue
                if func_id in s[pos[i]:]:
                    return True
            return False
        
        while True:
            # 所有节点都遍历过
            if len(all_func ^ finished_func) == 0: break
            for i,s in enumerate(raw_strategy):
                if pos[i] >= len(s):
                    continue
                j = pos[i]
                if set(self.G.predecessors(s[j])).issubset(finished_func):
                    pos[i] = j + 1
                    # if not is_exist(s[j]): # 只有在列表中不存在了，才添加？ 可能会死锁！
                    finished_func.add(s[j])
                    yield s[j],[i]
    
    # 按拓扑排序返回函数
    def topology_gen_strategy(self, raw_strategy, order=None):
        def find_pos(func):
            res = []
            for i,s in enumerate(raw_strategy):
                if func in s:
                    res.append(i)
                if func == self.source or func == self.sink:
                    return [-1]
            return res
        func = list(nx.topological_sort(self.G))
        for f in func:
            yield f,find_pos(f)

    # 用户自定义策略
    def custom_gen_strategy(self, raw_strategy, order=None):
        assert order != None, "custom generation policies must specify order"
        def find_pos(func):
            res = []
            for i,s in enumerate(raw_strategy):
                if func in s:
                    res.append(i)
                if func == self.source or func == self.sink:
                    return [-1]
            return res
        for f in order:
            yield f,find_pos(f)


class SequencingStrategy(metaclass=ABCMeta):
    def __init__(self, gen, raw_strategy, order, executor) -> None:
        self.raw_strategy = raw_strategy
        self.order = order
        self.executor = executor
        self.G = executor.G
        self.servers = executor.servers
        self.funcs = executor.funcs
        self.layers = executor.layers

        self.gen_strategy = GenStrategy(executor).get_gen_strategy(gen)

    '''
        server_id需要下载func_ids
        其中信息可以从servers, funcs, layers中获取(在self中)
    '''
    @abstractmethod
    def get_sequencing(self, server_id):
        pass

    # 根据gen_Strategy(调度策略)，生成每个核执行任务的序列
    def get_core_execution_sequence(self, server_id):
        st = self.executor.cluster.get_start_core_number(server_id)
        size = self.executor.cluster.get_core_number(server_id)

        seq_strategy = [[] for _ in range(self.executor.cluster.get_total_core_number())]
        for func_id,core_ids  in self.gen_strategy(self.raw_strategy, self.order):
            for core_id in core_ids:
                seq_strategy[core_id].append(func_id)

        return seq_strategy[st:st+size]
    
    # 获取需要下载的layer的集合
    def get_need_download_layers(self, server_id):
        st = self.executor.cluster.get_start_core_number(server_id)
        size = self.executor.cluster.get_core_number(server_id)
        layer_set = set() # 已经下载的集合
        for func_id, core_ids in self.gen_strategy(self.raw_strategy, self.order):
            for core_id in core_ids:
                if core_id >= st and core_id < st+size:
                    # 部署任务
                    tmp = set(self.funcs[func_id].layer) # 当前的集合
                    layer_set = layer_set | tmp
        return list(layer_set)
    
    def get_need_funcs(self, server_id):
        st = self.executor.cluster.get_start_core_number(server_id)
        size = self.executor.cluster.get_core_number(server_id)
        func_ids = []
        for func_id, core_ids in self.gen_strategy(self.raw_strategy, self.order):
            for core_id in core_ids:
                if core_id >= st and core_id < st+size:
                    func_ids.append(func_id)
        return func_ids
    
    def get_layers_size(self, func_ids):
        layer_set = set()
        for func_id in func_ids:
            layer_set = layer_set | set(self.funcs[func_id].layer)
        res = 0
        for layer in layer_set:
            res += self.layers[layer].size
        return res
    
    # 根据func_ids序列进行下载
    def get_sequencing_by_func_ids(self, func_ids):
        
        layer_set = set()
        layer_download_seq = []
        for func_id in func_ids:
            # 部署任务
            tmp = set(self.funcs[func_id].layer) # 当前的集合
            layer_download_seq.extend(tmp-layer_set) # 增量添加
            layer_set = layer_set | tmp
        # print(layer_download_seq)
        return layer_download_seq
            
    
# 根据部署的顺序下载镜像  
class FCFSSequencingBak(SequencingStrategy):
    def __init__(self, seq, raw_strategy, order, executor):
        super().__init__(seq, raw_strategy, order, executor)

    def get_sequencing(self, server_id):
        st = self.executor.cluster.get_start_core_number(server_id)
        size = self.executor.cluster.get_core_number(server_id)
        layer_download_seq = []
        layer_set = set() # 已经下载的集合
        for func_id, core_ids in self.gen_strategy(self.raw_strategy, self.order):
            for core_id in core_ids:
                if core_id >= st and core_id < st+size:
                    # 部署任务
                    tmp = set(self.funcs[func_id].layer) # 当前的集合
                    layer_download_seq.extend(tmp-layer_set) # 增量添加
                    layer_set = layer_set | tmp
        # print(layer_download_seq)
        return layer_download_seq

class FCFSSequencing(SequencingStrategy):
    def __init__(self, seq, raw_strategy, order, executor):
        super().__init__(seq, raw_strategy, order, executor)

    def get_sequencing(self, server_id):
        func_ids = [] 
        for funcs in self.get_core_execution_sequence(server_id):
            func_ids.extend(funcs)
        return self.get_sequencing_by_func_ids(func_ids)

# 使用sindey decomposition
class GLSASequencing(SequencingStrategy):
    def __init__(self, seq, raw_strategy, order, executor):
        super().__init__(seq, raw_strategy, order, executor)

    def get_sequencing(self, server_id):
        func_ids = self.get_need_funcs(server_id)
        download_sequence = Lcaa(self.executor, server_id).deploy_container_by_glsa(func_ids)
        print(download_sequence)
        return download_sequence

# 对"权重"最大镜像块先下载
class LOPOSequencing(SequencingStrategy):
    def __init__(self, seq, raw_strategy, order, executor):
        super().__init__(seq, raw_strategy, order, executor)

    def get_sequencing(self, server_id):
        st = self.executor.cluster.get_start_core_number(server_id)
        size = self.executor.cluster.get_core_number(server_id)
        func_ids = []
        for func_id, core_ids in self.gen_strategy(self.raw_strategy, self.order):
            for core_id in core_ids:
                if core_id >= st and core_id < st+size:
                    func_ids.append(func_id)

        layer_score = {}
        for func_id in func_ids:
            size = 0
            for layer_id in reversed(self.funcs[func_id].layer):
                size += self.layers[layer_id].size
                if layer_id in layer_score:
                    layer_score[layer_id] = max(layer_score[layer_id], size)
                else:
                    layer_score[layer_id] = size

        download_sequencing = list(layer_score.keys())
        download_sequencing.sort(key=lambda layer_id: -layer_score[layer_id])
        return download_sequencing

# 先下载小的镜像块
class CNTRSequencing(SequencingStrategy):
    def __init__(self, seq, raw_strategy, order, executor):
        super().__init__(seq, raw_strategy, order, executor)

    def get_sequencing(self, server_id):
        need_layer = self.get_need_download_layers(server_id)
        need_layer.sort(key=lambda layer_id: self.layers[layer_id].size)
        return need_layer
    

class DALPSequencingBak(SequencingStrategy):
    def __init__(self, seq, raw_strategy, order, executor):
        super().__init__(seq, raw_strategy, order, executor)

    def get_cost(self, x):
        # return 0
        # 需要下载的镜像大小 + 需要往外面传送数据的大小
        total_layer_size = self.get_layers_size(x)
        total_trans_size = 0
        for func_id in x:
            for sfid in list(self.G.successors(func_id)):
                if sfid not in self.need_funcs:
                    total_trans_size += self.executor.get_weight(func_id, sfid)

        alpha = 0
        layer_time = total_layer_size * self.servers[self.server_id].download_latency
        trans_time = total_trans_size * np.mean(self.executor.server_comm[self.server_id][self.executor.server_comm[self.server_id]!=0])
        score =  alpha*layer_time + (1-alpha)*trans_time
        return score

    def get_sequencing(self, server_id):
        self.server_id = server_id
        self.need_funcs = self.get_need_funcs(server_id)
        core_task = self.get_core_execution_sequence(server_id)
        # print(core_task)
        assert len(core_task) == self.executor.servers[server_id].core
        core_task = [i for i in core_task if len(i)!=0]
        
        func_ids = []
        while core_task:
            max_array = max(core_task, key=self.get_cost)
            func_ids.append(max_array[0])
            max_array.pop(0)
            if not max_array:
                core_task.remove(max_array)

        # print(func_ids)
        layer_download_seq = []
        layer_set = set()
        for func_id in func_ids:
            # 部署任务
            tmp = set(self.funcs[func_id].layer) # 当前的集合
            layer_download_seq.extend(tmp-layer_set) # 增量添加
            layer_set = layer_set | tmp
        return layer_download_seq



'''
1. 先根据每个核的任务调度序列，创建如下图，横轴表示核调度序列，纵轴表示任务之间的先序关系
o -> o -> o -> o -> ... -> o
  \,
o -> o -> o -> ... -> o
        /'
o -> o -> o

2. 计算每个节点node的权重
w(node) = 外部通信*平均通信延迟 + 执行时间 + max(child weight)

3. 根据权重从大到小下载每个节点的镜像块
'''
class DALPSequencing(SequencingStrategy):
    def __init__(self, seq, raw_strategy, order, executor):
        super().__init__(seq, raw_strategy, order, executor)

    def get_outside_comm(self, func_id):
        max_trans_size = 0
        for sfid in list(self.G.successors(func_id)):
            if sfid not in self.need_funcs:
                max_trans_size = max(max_trans_size, self.executor.get_weight(func_id, sfid))     
        return max_trans_size
    
    def get_func_seq(self, server_id, core_tasks):
        core_tasks_id = []
        id = 0
        id2task = {}
        # 编号
        for tasks in core_tasks:
            tasks_id = []
            for task in tasks:
                id2task[id] = task
                tasks_id.append(id)
                id+=1
            core_tasks_id.append(tasks_id)

        # 创建节点，一共id个
        G = nx.DiGraph()
        for i in range(id):
            G.add_node(i)
        
        # 核内部任务的序列关系
        for sublist in core_tasks_id:
            for i in range(len(sublist) - 1):
                G.add_edge(sublist[i], sublist[i+1])

        # 核之间任务的关系
        for i,s1 in enumerate(core_tasks_id):
            for j,s2 in enumerate(core_tasks_id):
                if i == j:
                    continue
                for t1 in s1:
                    for t2 in s2:
                        if nx.has_path(self.G, id2task[t1], id2task[t2]):
                            G.add_edge(t1, t2)
        
        # 平均通信延迟        
        mean_comm = np.mean(self.executor.server_comm[server_id][self.executor.server_comm[server_id]!=0])

        # 计算weight
        nodes = list(reversed(list(nx.topological_sort(G))))
        weight_node = {}
        for node in nodes:
            weight_node[node] = self.get_outside_comm(id2task[node]) * mean_comm + self.executor.func_process[id2task[node]][server_id]

            succ = G.successors(node)
            max_succ_weight = 0
            for suc in succ:
                max_succ_weight = max(max_succ_weight, weight_node[suc])

            weight_node[node] += max_succ_weight


        # node根据weight降序排序        
        nodes = sorted(nodes, key=lambda x: weight_node[x], reverse=True)

        res = [id2task[id] for id in nodes]
        return res

    def get_sequencing(self, server_id):
        self.server_id = server_id
        self.need_funcs = self.get_need_funcs(server_id)
        core_tasks = self.get_core_execution_sequence(server_id)
        # print(core_task)
        assert len(core_tasks) == self.executor.servers[server_id].core
        
        func_ids = self.get_func_seq(server_id, core_tasks)
        return self.get_sequencing_by_func_ids(func_ids)


class Sequencing:
    FCFS = 1
    GLSA = 2
    LOPO = 3
    CNTR = 4
    DALP = 5 # depentent aware layer pull

    seq_str = {"FCFS":FCFS, "GLSA":GLSA, "LOPO":LOPO, "CNTR":CNTR, "DALP": DALP}

    def __init__(self, seq, gen, raw_strategy, order, executor):
        self.seq = seq
        self.gen = gen
        self.raw_strategy = raw_strategy
        self.order = order
        self.executor = executor
        
    
    def get_sequencing_strategy(self, seq):
        match(seq):
            case Sequencing.FCFS:
                return FCFSSequencing
            case Sequencing.GLSA:
                return GLSASequencing
            case Sequencing.LOPO:
                return LOPOSequencing
            case Sequencing.CNTR:
                return CNTRSequencing
            case Sequencing.DALP:
                return DALPSequencing
            case _:
                assert False, "gen_strategy error"

    # def get_sequencing_from_str(self, seq_str):
    #     return self.get_sequencing_strategy(Sequencing.seq_str[seq_str])

    def get_download_sequence(self):
        sequencing = []
        sequencing_strategy = self.get_sequencing_strategy(self.seq)
        for i in range(self.executor.K):
            sequencing.append(sequencing_strategy(self.gen, self.raw_strategy, self.order, self.executor).get_sequencing(i))
        return sequencing
        
    
