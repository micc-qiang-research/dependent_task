from abc import abstractmethod, ABCMeta
import numpy as np
from cluster import Cluster
import math
from util import *
from .executor import Executor

class Scheduler(Executor,metaclass=ABCMeta):

    def __init__(self,data,config):
        super().__init__(data, config)
        self.data = data
        self.config = config

    '''
    功能：执行调度决策
    最后一行调用output_scheduler_strategy转成特定格式
    return self.output_scheduler_strategy()
    '''
    @abstractmethod
    def schedule(self):
        pass    

    '''
    功能：转换成统一格式
    返回： replica, place, download_sequence, gen_strategy=Executor.DUMB
    replica: 是否允许复制
    place: 每个core上的task
    download_sequence: 下载顺序
    gen_strategy: 生成策略,即如何生成任务部署次序
                  DUMB: 按照download_sequence次序，调用者需保证place的次序
                  TOPOLOGY: 按照拓扑排序
    '''
    @abstractmethod
    def output_scheduler_strategy(self):
        pass
