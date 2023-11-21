class SchedStrategy:
    def __init__(self, func, N):
        self.edge = False
        self.cloud = False
        self.user = False
        self.func = func # 记录是哪个函数的策略
        self.N = N # 函数个数

    def deploy_in_edge(self, server, core = None, \
        t_download_start = None, t_download_end = None, \
        t_prepare_start = None,  t_prepare_end = None, \
        t_execute_start = None, t_execute_end = None):
        self.edge = True
        self.edge_param = {
            "id": server,
            "core": core,
            "t_download_start": t_download_start,
            "t_download_end": t_download_end,
            "t_prepare_start": t_prepare_start,
            "t_prepare_end": t_prepare_end,
            "t_execute_start": t_execute_start,
            "t_execute_end": t_execute_end,
        }
    
    def clear_edge_deploy(self):
        self.edge = False

    def deploy_in_cloud(self, t_start, t_end):
        self.cloud = True
        self.cloud_param = {
            "start": t_start,
            "end": t_end
        }

    def clear_cloud_deploy(self):
        self.cloud = False

    def get_edge_id(self):
        if not self.edge:
            # assert False, "no edge deploy"
            raise Exception("no edge deploy")
        return self.edge_param["id"]

    def get_edge_start(self):
        if not self.edge:
            raise Exception("no edge deploy")
        return self.edge_param["t_execute_start"]

    def get_edge_end(self):
        if not self.edge:
            raise Exception("no edge deploy")
        return self.edge_param["t_execute_end"]

    def get_cloud_start(self):
        if not self.cloud:
           raise Exception("no cloud deploy")
        return self.cloud_param["start"]

    def get_cloud_end(self):
        if not self.cloud:
            raise Exception("no cloud deploy")
        return self.cloud_param["end"]

    def deploy_in_user(self, start, end):
        assert self.func == 0 or self.func == self.N-1, "not source or sink"
        self.user = True
        self.user_param = {
            "start": start,
            "end": end
        }
        assert start == end, "user deploy error"

    def get_user_end(self):
        if not self.user:
            assert False, "no user deploy"
        return self.user_param["end"]

    def debug(self):
        print("*"*20)
        print("func: " + str(self.func))
        if self.edge:
            print("edge: " + str(self.edge_param))
        if self.cloud:
            print("cloud: " + str(self.cloud_param))
        if self.user:
            print("user: " + str(self.user_param))
    
    def debug_readable(self):
        from util import colors, prepare_color, download_color, user_color
        func = self.func
        if func != 0 and func != self.N - 1:
            func_color = colors[func - 1]
        bars = ""

        str_json = "{{\"row\": \"{}\", \"from\": {}, \"to\": {}, \"color\": \"{}\"}},"
        if self.edge:
            name = "edge_" + str(self.edge_param["id"]) + "_" + str(self.edge_param["core"])
            name_download = "edge_" + str(self.edge_param["id"]) + "_d"
            # download 
            bars = bars + str_json.format(name_download, self.edge_param["t_download_start"], self.edge_param["t_download_end"], download_color)

            # prepare
            bars = bars + str_json.format(name, self.edge_param["t_prepare_start"], self.edge_param["t_prepare_end"], prepare_color)
            
            # exec
            bars = bars + str_json.format(name, self.edge_param["t_execute_start"], self.edge_param["t_execute_end"], func_color)
        if self.cloud:
            bars = bars + str_json.format("cloud", self.cloud_param["start"], self.cloud_param["end"], func_color)
        if self.user:
            bars = bars + str_json.format("user", self.user_param["start"], self.user_param["end"]+1e-2, user_color)
        
        return bars
