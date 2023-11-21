import portion as P


class Core:
    def __init__(self, idx):
        self.idx = idx
        self.interval = P.closedopen(0, P.inf)

    def occupy(self, start, end):
        i = P.closedopen(start, end)
        if not self.interval.contains(i):
            print(self.interval)
            print(start, end)
            assert False, "occupy error"

        self.interval = self.interval - P.closedopen(start, end)
    
    def release(self, start, end):
        i = P.closedopen(start, end)
        if not (self.interval & i).empty:
            assert False, "release error" # 释放的是已经占据的
        self.interval = self.interval | i

    def __repr__(self):
        return self.interval.__str__()

    def __str__(self):
        return self.interval.__str__()

    def __iter__(self):
        return self.interval.__iter__()
        
class EdgeServer:
    def __init__(self, idx, core):
        self.idx = idx
        self.cores = [Core(i) for i in range(core)]
        self.download_complete = 0

    # 在某个核上查找任务最早开始时间
    def __core_ESTfind(self, t, t_prepare, t_execute, core):
        for i in core:
            if i.upper >= t_execute + max(t, i.lower + t_prepare):
                return max(t, i.lower+t_prepare)

    # 放置在此server最早的开始时间
    def ESTfind(self, t, t_prepare, t_execute):
        start_time = P.inf
        early_core = None
        for core in self.cores:
            res = self.__core_ESTfind(t, t_prepare, t_execute, core)
            if res < start_time:
                start_time = res
                early_core = core
        return early_core, start_time

    def place(self, core, start, end):
        idx = core.idx
        print(f"edge[{self.idx}-{idx}] occupy: {start}-{end}")
        self.cores[idx].occupy(start, end)

    def release(self, core_id, start, end):
        print(f"edge[{self.idx}-{core_id}] release: {start}-{end}")
        self.cores[core_id].release(start, end)