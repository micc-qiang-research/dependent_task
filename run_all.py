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
from types import SimpleNamespace

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

    def get_scheduler_bak():
        res = []
        if run_config.sequence:
            sequence_strategy = run_config.sequence_strategy
            sequence_scheduler = run_config.sequence_scheduler
            for sched in sequence_scheduler:
                for seq in sequence_strategy:
                    res.append((f"{sched}-{seq}", sched, seq))
        else:
            scheduler = run_config.scheduler_run
            for sched in scheduler:
                res.append((f"{sched}", sched, None))
        return res
    
    def get_scheduler():
        res = []
        # scheduler
        if run_config.setting == 0:
            scheduler = run_config.scheduler
            for sched in scheduler:
                s = getattr(run_config, sched)
                res.append((f"{sched}", s[0], s[1]))
        # deploy
        elif run_config.setting == 1:
            deploy = run_config.deploy
            default_sequence = run_config.default_sequence
            for d in deploy:
                res.append((f"{d}-{default_sequence}", d, default_sequence))
        else:
            sequence = run_config.sequence
            default_deploy = run_config.default_deploy
            for s in sequence:
                res.append((f"{default_deploy}-{s}", default_deploy, s))
        return res

    def run_once(filename):
        
        # val = filename.split('/')[2].split('.json')[0].split('_')
        param = dict(zip(RunExperiment.keys, RunExperiment.dataset_params))
        try:
            data = DataByJson(filename) # read Data
            # scheduler = run_config.scheduler_run
            scheduler = RunExperiment.get_scheduler()
            for name,sch,seq in scheduler:
                sched = eval(sch)
                config = None
                if seq:
                    config = SimpleNamespace()
                    config.sequence = seq
                param[name] = Analysis(data, *sched(data, config).schedule(), config=config).summarize()
            
            # 每一条数据是一个dict，含有如下字段：['n', 'fat', 'density', 'regularity', 'jump', 'makespan_xx']
        except:
            print("Error occure")
            exit(1)
        return param

    # 增量保存 **废弃了**
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

# 临时改变名字，trival
def rename(k, ccr, lfr, dcr):
    file = f"__result__/app30-放置算法对比/data_{k}_{ccr}_{lfr}_{dcr}.pkl"
    with open(file, 'rb') as handle:
        df = pickle.load(handle)
        # df.rename(columns={'Propose-GLSA': 'LPTS-GLSA'}, inplace=True)
        # df.rename(columns={'Propose-FCFS': 'LPTS-FCFS'}, inplace=True)
        # df.rename(columns={'Propose-LOPO': 'LPTS-LOPO'}, inplace=True)
        # df.rename(columns={'Propose-CNTR': 'LPTS-CNTR'}, inplace=True)
        # df.rename(columns={'Propose-DALP': 'LPTS-DALP'}, inplace=True)
        # df.rename(columns={'makespan_SDTS': 'SDTS-DALP'}, inplace=True)
        # df.rename(columns={'makespan_Propose': 'LPTS-DALP'}, inplace=True)
        # df.rename(columns={'makespan_GenDoc': 'GenDoc-DALP'}, inplace=True)
        # df.rename(columns={'makespan_HEFT': 'HEFT-DALP'}, inplace=True)
        # df.rename(columns={'makespan_LCAAP': 'LCAAP-DALP'}, inplace=True)
        df.rename(columns={'SDTSPlus': 'SDTSPlus-DALP'}, inplace=True)

    with open(file, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
                    # rename
                    # rename(k ,ccr, lfr, dcr)
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
