from .SidneyDecomposition import SidneyDecomposition

class Job:
    def __init__(self, is_container_job, data):
        self.is_container_job = is_container_job
        self.data = data
    
    def is_container_job(self):
        return self.is_container_job

class Lcaa:
    def __init__(self, executor, server_id):
        self.executor = executor
        self.bandwidth = 1 / executor.servers[server_id].download_latency
        self.layers = set()

    # 添加一个job
    def add_job(self, is_container, data):
        if not hasattr(self, "jobs"):
            self.jobs = []
        
        job = Job(is_container, data)
        self.jobs.append(job)
        return len(self.jobs) - 1 # 返回添加的任务的index
    
    def get_job_by_index(self, index):
        return self.jobs[index]

    def get_job_number(self):
        return len(self.jobs)
    
    def deploy_container(self, image):
        res = self.get_icrement_layer_digest(image) - self.layers
        self.layers = self.layers | self.get_icrement_layer_digest(image)
        return sorted(list(res))
    
    def get_icrement_layer_digest(self, image):
        layers = set(self.executor.funcs[image].layer)
        return layers - self.layers

    # 此server添加image后，增加的容器容量
    def get_icrement(self, image):
        layers = self.executor.funcs[image].layer      
        increment = 0
        for layer_id in layers:
            if layer_id not in self.layers:
                increment += self.executor.layers[layer_id].size
        return increment

    def convert_to_pred_job_seq_problem(self, images):
        layers = {}
        W = []
        P = []
        dag = set()
        # lower_bound_p = 1 / self.bandwidth
        for image in images:
            cid = self.add_job(True, image)
            W.append(1)
            P.append(0)
            # for layer in self.data.get_image_layers(image):
            for layer_id in self.executor.funcs[image].layer:
                layer = self.executor.layers[layer_id]
                if layer_id not in layers:
                    lid = self.add_job(False, layer_id)
                    W.append(0)
                    P.append(layer.size / self.bandwidth)
                    layers[layer_id] = lid # 记录了实际的编号
                else:
                    lid = layers[layer_id]
                dag.add((lid, cid))
        # P = [lower_bound_p if p == 0 else p for p in P]
        return self.get_job_number(), dag, W, P

    def glsa(self, images):
        n, dag, W, P = self.convert_to_pred_job_seq_problem(images)
        Y = SidneyDecomposition(n, dag, W, P).run()
        startup_latency = []
        L_seq = []
        layer_set = set()
        for S in Y:
            S_c = []
            S_l_digest = []
            for j in S:
                job = self.get_job_by_index(j)
                if job.is_container_job:
                    S_c.append(job.data)
                else:
                    S_l_digest.append(job.data)
                    layer_set.add(job.data)

            # 一个一个容器开始部署
            while len(S_c) > 0:
                c_min = None
                layer_fetch = 0
                # 选择一个容器，需要下载的镜像块最少
                for c in S_c:
                    l_inc = self.get_icrement(c)
                    l_inc_digest = self.get_icrement_layer_digest(c)
                    assert l_inc_digest.issubset(layer_set), "l_inc_digest is not subset of S_l_digest"
                    if c_min == None or l_inc < layer_fetch:
                        c_min = c
                        layer_fetch = l_inc

                S_c.remove(c_min)
                # 对容器c_min进行部署
                L_seq.extend(self.deploy_container(c_min))
        
        return L_seq

    def deploy_container_by_glsa(self, images):
        if len(images) == 0:
            return []
        return self.glsa(images)
