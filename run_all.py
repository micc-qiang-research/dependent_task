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
from generate_dataset import traverse_and_build

# 获取生成的所有数据集，然后执行一次实验
class RunExperiment:
    
    # keys = ['n', 'fat', 'density', 'regularity', 'jump']
    keys = ['k', 'ccr', 'lfr', 'dcr']

    # 开启的线程个数
    cpu_cnt = 16

    # 将标准输出重定向
    def redirect_stdout_stderr(out, err):
        # 将标准输出重定向到文件
        sys.stdout = open(out, 'w')
        # 将标准错误重定向到文件
        # sys.stderr = open(err, 'w')

    # 恢复标准输出
    def recover_stdout_stderr():
        sys.stdout.close()
        # sys.stderr.close()
        sys.stdout = sys.__stdout__
        # sys.stderr = sys.__stderr__

    # 对某一个filename执行各个调度器测试
    def run_once_wrapper(filename):
        print("start exec ", filename)
        param = RunExperiment.run_once(filename)
        return param

    def run_once(filename):
        
        # val = filename.split('/')[2].split('.json')[0].split('_')
        param = dict(zip(RunExperiment.keys, RunExperiment.dataset_params))
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
        print(filename , " saved!")

    # dataset_params = [k, ccr, lfr, dcr]
    def run(dataset_params):
        RunExperiment.dataset_params = dataset_params
        label = "_".join([str(i) for i in  dataset_params])

        # 获取所有的测试文件
        filenames = glob('data/json/*.json')
        data = []

        # print('Using {} cores'.format(RunExperiment.cpu_cnt))
        # 标准输出重定向到文件，避免干扰
        RunExperiment.redirect_stdout_stderr(get_in_result_path("run_all_out.txt"), get_in_result_path("run_all_err.txt"))

        pool = mp.Pool(RunExperiment.cpu_cnt)
        pbar = tqdm(total=len(filenames))

        # 创建线程池
        results = [pool.apply_async(RunExperiment.run_once_wrapper, (filename, ),callback=lambda _:pbar.update(1)) for filename in filenames]

        for result in results:
            data.append(result.get())

        # 恢复标准输出
        RunExperiment.recover_stdout_stderr()
        # print(data)

        # 保存结果
        RunExperiment.save("data_"+label+".pkl", data)


# 判断对应的数据集目录是否存在
def check_and_build(k ,ccr, lfr, dcr):
    import os
    import shutil

    gen_path = "data/json"
    save_path = "data/json_{}_{}_{}_{}".format(k, ccr, lfr, dcr)  # 文件夹路径
    
    # 删除原目录
    if os.path.exists(gen_path):
        shutil.rmtree(gen_path)
    
    if os.path.exists(save_path):
        shutil.copytree(save_path, gen_path)
        return True
    os.mkdir(gen_path)
    return False

def save_dataset(k ,ccr, lfr, dcr):
    import shutil
    gen_path = "data/json"
    save_path = "data/json_{}_{}_{}_{}".format(k, ccr, lfr, dcr)  # 文件夹路径
    shutil.copytree(gen_path, save_path)


def check_exist(k ,ccr, lfr, dcr):
    import os
    return os.path.exists("__result__/data_{}_{}_{}_{}.pkl".format(k ,ccr, lfr, dcr))

if __name__ == '__main__':
    K = run_config.range_K # server number
    CCRs = run_config.range_ccr # commucation to computation ratio
    LFRs = run_config.range_lfr # Layer number to function number ratio
    DCRs = run_config.range_dcr # download to computation ratio

    # K = [5] # server number
    # CCRs = [0.5] # commucation to computation ratio
    # LFRs = [5.0] # Layer number to function number ratio
    # DCRs = [2.0] # download to computation ratio

    cnt = 0
    for k in K:
        for ccr in CCRs:
            for lfr in LFRs:
                for dcr in DCRs:
                    data = [k, ccr, lfr, dcr]
                    cnt += 1
                    print(f"==========={cnt}/{len(K)*len(CCRs)*len(LFRs) * len(DCRs)}===============")
                    print("k: {}, ccr: {}, lfr: {}, dcr: {}".format(*data))
                    if check_exist(*data):
                        print("using cache!")
                        continue
                    print("building data...")
                    if(not check_and_build(*data)):
                        traverse_and_build(*data)
                        save_dataset(*data)
                    else:
                        print("!read from baking")
                    print("test algorithm...")
                    RunExperiment.run(data)
