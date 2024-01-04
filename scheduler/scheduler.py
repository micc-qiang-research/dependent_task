from abc import abstractmethod, ABCMeta
import numpy as np
from cluster import Cluster
import math
from util import *
from .executor import Executor

class Scheduler(Executor,metaclass=ABCMeta):

    def __init__(self,data,config):
        super().__init__(data)
        self.data = data
        self.config = config

    @abstractmethod
    def schedule(self):
        pass

    def showGantt(self, name):
        bars = ""
        for s in self.strategy:
            bars = bars + s.debug_readable(self.cluster)
        output_gantt_json(name, self.cluster.get_names(), bars[:-1], self.gs(self.sink).get_user_end())
        draw_gantt()

    def get_total_time(self):
        return self.gs(self.sink).get_edge_end()

    def show_result(self, name):
        print("total time: {:.2f}".format(self.get_total_time()))
        if self.config.gantta:
            self.showGantt(name)