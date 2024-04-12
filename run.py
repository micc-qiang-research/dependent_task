
from dataloader import *
from scheduler import *
from util import *
import argparse
from types import SimpleNamespace
import os
from analysis import Analysis
import logging

def parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-d","--data", type=str, default="./data/cluster/data_2.txt", help="specify data file")
    parser.add_argument("-d","--data", type=str, default="./data/task/application_csv/app_3.csv", help="specify data file")
    parser.add_argument("-s","--scheduler", type=str, choices=['SDTS','HEFT','GenDoc','Optim','LCAAP','SDTSPlus','Propose', 'ProposePlus'], default="SDTS", help="specify scheduler type")
    parser.add_argument("-g","--gantta", action="store_true", help="show gantta or not")
    parser.add_argument("--dag", action="store_true", help="show dag or not")
    args = parser.parse_args()
    return {
        "data": args.data,
        "scheduler": args.scheduler,
        "gantta": args.gantta,
        "dag": args.dag,
        "log_level": logging.DEBUG
    }

def getFileType(file):
    return file.split('.')[-1]


def getFileLoader(filetype):
    match(filetype):
        case "txt":
            return DataByTxt
        case "csv":
            return DataByCsv
        case "json":
            return DataByJson
        case _:
            assert False, "unsupport file type"
    

if __name__ == '__main__':
    config = SimpleNamespace(**parse())
    filetype = getFileType(config.data)
    print("data file: ", config.data)
    print("data type: ", filetype)
    print("scheduler: ", config.scheduler)
    # try:
    data : Data = getFileLoader(filetype)(config.data)
    # draw_dag(data.G)
    if config.dag:
        draw_dag(data.G)
    scheduler :Scheduler = eval(config.scheduler)(data,config)
    Analysis(data, *scheduler.schedule(), config=config).summarize()
    # except Exception as e:
    #     print(e)
    #     exit(1)
