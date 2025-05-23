import portion as P

# 按至少0.01的粒度occupy，否则会有数值问题
class Core:
    def __init__(self, idx):
        self.idx = idx
        self.interval = P.closedopen(0, P.inf)

    def round_2(self, start, end):
        start_tmp = round(start, 2)
        end_tmp = round(end, 2)
        if start < start_tmp:
            start = round(start_tmp - 0.01,2)
        else:
            start = start_tmp
        
        if end > end_tmp: # 舍入了
            end = round(end_tmp + 0.01,2)
        else:
            end = end_tmp
        return start, end

    def occupy(self, start, end):
        start, end = self.round_2(start, end)
        i = P.closedopen(start, end)
        if not self.interval.contains(i):
            print(self.interval)
            print(start, end)
            assert False, "occupy error"

        self.interval = self.interval - P.closedopen(start, end)
    
    def release(self, start, end):
        start, end = self.round_2(start, end)
        i = P.closedopen(start, end)
        if not (self.interval & i).empty:
            assert False, "release error" # 释放的是已经占据的
        self.interval = self.interval | i

    def is_occupy(self, start, end) -> bool:
        i = P.closedopen(start, end)
        return not self.interval.contains(i)
    
    def find_est(self, start, size) -> bool:
        start, size = self.round_2(start, size)
        for i in self.interval:
            if i.lower >= start and i.upper >= size + i.lower:
                return i.lower
        assert False, "never be there"


    def __repr__(self):
        return self.interval.__str__()

    def __str__(self):
        return self.interval.__str__()

    def __iter__(self):
        return self.interval.__iter__()
        
class Server:
    def __init__(self, idx, core_number):
        self.idx = idx
        self.cores = [Core(i) for i in range(core_number)]
        self.download_complete = 0

    # 在某个核上查找任务最早开始时间
    def ESTfindByCore(self, t, t_prepare, t_execute, core):
        for i in core:
            if i.upper >= t_execute + max(t, i.lower + t_prepare):
                return max(t, i.lower+t_prepare)

    # 放置在此server最早的开始时间
    def ESTfind(self, t, t_prepare, t_execute):
        start_time = P.inf
        early_core = None
        for core in self.cores:
            res = self.ESTfindByCore(t, t_prepare, t_execute, core)
            if res < start_time:
                start_time = res
                early_core = core
        return early_core, start_time

    def get_core(self, core_id):
        return self.cores[core_id]


    def place(self, core_id, start, end):
        # print(f"edge[{self.idx}-{core_id}] occupy: {start}-{end}")
        self.cores[core_id].occupy(start, end)

    def release(self, core_id, start, end):
        # print(f"edge[{self.idx}-{core_id}] release: {start}-{end}")
        self.cores[core_id].release(start, end)


class Cluster:
    def __init__(self, cores_number_array):
        self.K = len(cores_number_array)
        self.cores_number_array = cores_number_array
        self.servers = [Server(i, cores_number_array[i]) for i in range(self.K)]

        self.server_name = []
        for i in range(self.K):
            self.server_name.append(f"server_{i}_d")
            for j in range(cores_number_array[i]):
                self.server_name.append(f"server_{i}_{j}")

    def get_total_core_number(self):
        return sum(self.cores_number_array)
    
    def get_core_number(self, server_id):
        return self.cores_number_array[server_id]

    def get_start_core_number(self, server_id):
        return sum(self.cores_number_array[:server_id])
    
    '''
    根据给定的core_id获取对应的server id
    假设含有两个server: server 0、server 1, server 0 有2个core, server 1有2个core, 那么core的编号规则如下:
    server 0 core 0 -> 0
    server 0 core 1 -> 1
    server 1 core 0 -> 2
    server 1 core 1 -> 3
    '''
    def get_server_by_core_id(self, core_id):
        for i, core_number in enumerate(self.cores_number_array):
            if core_id < core_number:
                return i, core_id
            core_id -= core_number
        assert False, "core_id is too big"

    '''
    获取server_id:core_id在所有core编号规则下的id
    '''
    def get_total_core_id(self, server_id, core_id):
        res = 0
        for i in range(server_id):
            res += self.cores_number_array[i]
        return res + core_id
        

    def get_server(self):
        return self.servers

    def get_download_complete(self, server_id):
        return self.servers[server_id].download_complete

    def set_download_complete(self, server_id, value):
        self.servers[server_id].download_complete = value

    def place(self, server_id, core_id, start, end):
        self.servers[server_id].place(core_id, start, end)

    def release(self, server_id, core_id, start, end):
        self.servers[server_id].release(core_id, start, end)

    def get_core_EST(self, server_id, core_id, t_prepare, t_execute, t):
        return self.servers[server_id].ESTfindByCore(t, t_prepare, t_execute, self.get_server_core(server_id, core_id))

    def get_server_core(self, server_id, core_id):
        return self.servers[server_id].get_core(core_id)

    # def get_cloud_core_name(self, start, end):
    #     if not hasattr(self, "cloud_core_number"):
    #         self.cloud_core_number = 1
    #         self.cloud_core_names = ["cloud_0_0"]
    #         self.cloud_cores = [Core(0)]
        
    #     for i, core in enumerate(self.cloud_cores):
    #         if core.is_occupy(start, end):
    #             continue
    #         else:
    #             core.occupy(start, end)
    #             return self.cloud_core_names[i]
        
    #     # add new core
    #     self.cloud_core_names.append("cloud_0_" + str(self.cloud_core_number))
    #     self.cloud_cores.append(Core(self.cloud_core_number))
    #     self.cloud_core_number += 1
    #     self.cloud_cores[-1].occupy(start, end)
    #     return self.cloud_core_names[-1]

    def get_names(self):
        for i in range(self.K):
            yield f"server_{i}_d"
            for j in range(self.cores_number_array[i]):
                yield f"server_{i}_{j}"