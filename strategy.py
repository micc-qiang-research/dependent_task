class SchedStrategy:
    def __init__(self, func, N):
        self.edge = False
        self.cloud = False
        self.user = False
        self.func = func # 记录是哪个函数的策略
        self.N = N # 函数个数

    def deploy_in_edge(self, server, core = None, t_download_finish = None, t_start = None, t_end = None):
        self.edge = True
        self.edge_param = {
            "id": server,
            "core": core,
            "download_finish": t_download_finish,
            "start": t_start,
            "end": t_end
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
            assert False, "no edge deploy"
        return self.edge_param["start"]

    def get_edge_end(self):
        if not self.edge:
            assert False, "no edge deploy"
        return self.edge_param["end"]

    def get_cloud_start(self):
        if not self.cloud:
            assert False, "no cloud deploy"
        return self.cloud_param["start"]

    def get_cloud_end(self):
        if not self.cloud:
            assert False, "no cloud deploy"
        return self.cloud_param["end"]

    def deploy_in_user(self, start, end):
        assert self.func == 0 or self.func == self.N-1, "not source or sink"
        self.user = True
        self.user_param = {
            "start": start,
            "end": end
        }
        assert start == end, "user deploy error"

    def debug(self):
        print("*"*20)
        print("func: " + str(self.func))
        if self.edge:
            print("edge: " + str(self.edge_param))
        if self.cloud:
            print("cloud: " + str(self.cloud_param))
        if self.user:
            print("user: " + str(self.user_param))
        print("*"*20)
