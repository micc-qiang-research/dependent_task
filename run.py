
from data import Data
from scheduler.sdts import SDTS
from scheduler.heft import HEFT
from scheduler.scheduler import Scheduler

if __name__ == '__main__':
    data = Data("./data/data_3.txt")
    # draw_dag(data.G)
    scheduler :Scheduler = SDTS(data)
    # scheduler :Scheduler = HEFT(data)
    scheduler.schedule()