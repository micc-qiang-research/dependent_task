
from data import Data
from scheduler.sdts import SDTS
from scheduler.heft import HEFT
from scheduler.scheduler import Scheduler
from util import *

if __name__ == '__main__':
    data = Data("./data/data_2.txt")
    draw_dag(data.G)
    scheduler :Scheduler = SDTS(data)
    # scheduler :Scheduler = HEFT(data)
    scheduler.schedule()