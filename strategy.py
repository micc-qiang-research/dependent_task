class Strategy:
    def __init__(self, func, N):
        self.func = func # 记录是哪个函数的策略
        self.N = N # 函数个数
        self.deployment = {}
    
    def __get_a_deployment(self,server_id, core_id=None, t_execute_start=None, t_execute_end=None,t_download_start=0, t_download_end=0):
        return {
            "valid": True,
            "server_id": server_id,
            "core_id": core_id,
            "t_execute_start": t_execute_start,
            "t_execute_end": t_execute_end,
            "t_download_start": t_download_start,
            "t_download_end": t_download_end
        }

    def is_deploy(self, name):
        if name not in self.deployment or not self.deployment[name]['valid']:
            return False
        return True
        
    def clear(self, name):
        if self.is_deploy(name):
            self.deployment[name]['valid'] = False

    def deploy(self, name, server_id, core_id=None, t_execute_start=None, t_execute_end=None,t_download_start=0, t_download_end=0):
        self.deployment[name] = self.__get_a_deployment(server_id, core_id, t_execute_start, t_execute_end, t_download_start, t_download_end)

    def get_deploy_info(self, name):
        assert self.is_deploy(name), "not deploy"
        return self.deployment[name]
    
    def get_all_deploys(self):
        return list(self.deployment.values())
    
    def get_all_deploys_names(self):
        return list(self.deployment.keys())

    def get_name(self,server_id, core_id):
        return "server_" + str(server_id) + "_" + str(core_id)

    def get_download_name(self, server_id):
        return "server_" + str(server_id) + "_d"

    def debug_readable(self):
        from util import colors, prepare_color, download_color, user_color
        func = self.func
        if func != 0 and func != self.N - 1:
            if func - 1 >= len(colors)-3:
                assert False, "too many function"
            else:
                func_color = colors[func - 1]
        else:
            func_color = user_color
        bars = ""

        str_json = "{{\"row\": \"{}\", \"from\": {}, \"to\": {}, \"color\": \"{}\"}},"
        for deploy in self.get_all_deploys():
            name = self.get_name(deploy["server_id"], deploy["core_id"])
            name_download = self.get_download_name(deploy["server_id"])
            # download 
            bars = bars + str_json.format(name_download, deploy["t_download_start"], deploy["t_download_end"], download_color)
            
            # exec
            bars = bars + str_json.format(name, deploy["t_execute_start"], deploy["t_execute_end"], func_color)
        
        return bars