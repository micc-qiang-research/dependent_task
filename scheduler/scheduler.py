from abc import abstractmethod, ABCMeta

class Scheduler(metaclass=ABCMeta):

    @abstractmethod
    def schedule(self):
        pass