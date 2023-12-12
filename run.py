
from data import Data
from scheduler.sdts import SDTS
from scheduler.heft import HEFT
from scheduler.scheduler import Scheduler
from util import *
import argparse
from types import SimpleNamespace

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data", type=str, default="./data/cluster/data_2.txt", help="specify data file")
    parser.add_argument("-s","--scheduler", type=str, choices=['SDTS','HEFT'], default="SDTS", help="specify scheduler type")
    args = parser.parse_args()
    return {
        "data": args.data,
        "scheduler": args.scheduler
    }

if __name__ == '__main__':
    config = SimpleNamespace(**parse())
    print("data file : ", config.data)
    print("scheduler: ", config.scheduler)
    try:
        data = Data(config.data)
        draw_dag(data.G)
        scheduler :Scheduler = eval(config.scheduler)(data)
        scheduler.schedule()
    except Exception as e:
        print(e)
        exit(1)
