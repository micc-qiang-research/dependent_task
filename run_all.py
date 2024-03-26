from dataloader import *
from scheduler import *
from util import *
from analysis import Analysis

import os
import pandas as pd
import pickle
import multiprocessing as mp
from glob import glob
from util import get_in_result_path
from config import run_config
import sys
import concurrent.futures
from tqdm import tqdm


def __redirect_stdout_stderr(out, err):
    # 将标准输出重定向到文件
    sys.stdout = open(out, 'w')
    # 将标准错误重定向到文件
    # sys.stderr = open(err, 'w')

def __recover_stdout_stderr():
    sys.stdout.close()
    # sys.stderr.close()
    sys.stdout = sys.__stdout__
    # sys.stderr = sys.__stderr__


def run(pack):
    # 文件编号，文件名
    progree_bar,filename = pack
    print("start exec ", filename)
    param = run_one(filename)
    progree_bar.update(1)
    return param


def run_one(filename):
    val = filename.split('/')[2].split('.json')[0].split('_')
    param = dict(zip(keys, val))
    try:
        data = DataByJson(filename) # read Data
        scheduler = run_config.scheduler_run
        for s in scheduler:
            sched = eval(s)
            param['makespan_'+s] = Analysis(data, *sched(data, None).schedule()).summarize()
        
        # 每一条数据是一个dict，含有如下字段：['n', 'fat', 'density', 'regularity', 'jump', 'makespan_xx']
    except:
        print("Error occure")
        exit(1)
    return param

# 增量保存
def save(filename, data):
    result_filename = get_in_result_path(filename)
    if os.path.exists(result_filename):
        df = pd.DataFrame(data)
        with open(result_filename, 'rb') as handle:
            df_old = pickle.load(handle)
            df = pd.DataFrame(data)
            if df_old.shape[0] == len(data):
                makespan_keys = [k for k in data[0] if k.startswith('makespan_')]
                other_keys = [k for k in data[0] if not k.startswith('makespan_')]

                tmp = df_old.merge(df, on=other_keys, how="inner")
                for k in makespan_keys:
                    if k in df_old:
                        df_old[k] = tmp[k+"_y"]
                    else:
                        df_old[k] = tmp[k]
                df = df_old
                
    else:
        df = pd.DataFrame(data)
    
    with open(result_filename, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Data saved!")


# n_trials = 5
# minalpha = 20
# maxalpha = 150
# n = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400]
# fat = [0.1, 0.4, 0.8]
# density = [0.2, 0.8]
# regularity = [0.2, 0.8]
# jump = [1,2,4]
# ccr = [0.1, 0.25, 0.5, 0.8, 1, 2, 5, 8, 10, 15, 20, 25, 30]
# b = [0.1, 0.2, 0.5, 0.75, 1, 2]
# p = [4,8,16,32]

n_trials = 1
batch_number = 4

keys = ['n', 'fat', 'density', 'regularity', 'jump']

# 获取所有的测试文件
filenames = glob('data/json/*.json')

# 开启的线程个数
cpu_cnt = 4
print('Using {} cores'.format(cpu_cnt))

# columns = ['n', 'fat', 'density', 'regularity', 'jump', 'ccr','b','p', 'makespan_HEFT', 'makespan_PSO', 'makespan_IPEFT']


data = []

__redirect_stdout_stderr(get_in_result_path("run_all_out.txt"), get_in_result_path("run_all_err.txt"))
# 创建线程池
with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_cnt) as executor:

    with tqdm(total=len(filenames), desc='Progress', ncols=80) as progress_bar:
        # 提交任务给线程池
        futures = [executor.submit(run, (progress_bar, filename)) for filename in filenames]
        # 获取任务的返回结果
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            data.append(result)

__recover_stdout_stderr()
print(data)

# 保存结果
save("data.pkl", data)