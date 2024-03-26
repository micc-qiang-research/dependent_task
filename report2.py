import pandas as pd
import pickle
from util import prepare_draw_cdf, draw_cdf, get_in_result_path, draw_pairwise_comparison_table
import numpy as np
from config import run_config
from glob import glob
import re

'''
文件系统的格式为: data_k_ccr_lfr_dcr.pkl
若是对ccr做敏感性分析, 则保持其他参数不变, 提取ccr变量
'''

default_k = 5 
default_ccr = 0.5 
default_lfr = 2
default_dcr = 5

# 获取所有ccr文件
def get_all_files_ccr():
    filenames = glob("__result__/data_"+str(default_k)+"_*_"+str(default_lfr)+"_"+str(default_dcr)+".pkl")
    return filenames

def get_all_files_k():
    filenames = glob("__result__/data_*_"+str(default_ccr)+"_"+str(default_lfr)+"_"+str(default_dcr)+".pkl")
    return filenames

def get_all_files_dcr():
    filenames = glob("__result__/data_"+str(default_k)+"_"+str(default_ccr)+"_"+str(default_lfr)+"_*.pkl")
    return filenames

def get_all_files_lfr():
    filenames = glob("__result__/data_"+str(default_k)+"_"+str(default_ccr)+"_*_"+str(default_dcr)+".pkl")
    return filenames

class Type:
    K = 0
    CCR = 1
    LFR = 2
    DCR = 3

# 获得匹配的文件所有取值
def get_index(ltype, filenames):
    res = []

    pattern_k = r'__result__/data_(.*?)_{}_.*'.format(default_ccr)
    pattern_ccr = r'__result__/data_{}_(.*?)_.*'.format(default_k)
    pattern_lfr = r'__result__/data_{}_{}_(.*?)_.*'.format(default_k, default_ccr)
    pattern_dcr = r'__result__/data_{}_{}_{}_(.*?).pkl'.format(default_k, default_ccr, default_lfr)
    patterns = [pattern_k, pattern_ccr, pattern_lfr, pattern_dcr]
    pattern = patterns[ltype]

    for file_name in filenames:
        # 使用正则表达式匹配标记部分
        match = re.search(pattern, file_name)
        if match:
            marked_part = match.group(1)
            # 打印标记部分
            res.append(float(marked_part))
    return res
    



'''
    返回格式：一个字典 + ccr取值
    dict:
        {
        "SDTS": [xx, xx, xx],
        "GenDoc": [xx, xx, xx],
        ...
        }
    ccr value:
        [0.1, 0.5, 1.0, 1.5, 2.0]
'''
def get_makespan_ccr():
    filenames = get_all_files_ccr()
    print(filenames)
    print(get_index(Type.K, filenames))
    print(get_index(Type.CCR, filenames))
    print(get_index(Type.LFR, filenames))
    print(get_index(Type.DCR, filenames))


get_makespan_ccr()


# with open(get_in_result_path('data.pkl'), 'rb') as handle:
#     data = pickle.load(handle)
#     # scheduler = ["SDTS","GenDoc", "HEFT", "Optim"]
#     scheduler = run_config.scheduler_show
#     makespan = []
#     for s in scheduler:
#         makespan.append(list(data["makespan_"+s]))

#     prepare_draw_cdf(makespan)
#     draw_cdf(len(scheduler), scheduler, "lat_cdf.csv")

#     result_better = np.zeros((len(scheduler), len(scheduler)))
#     result_equal = result_better.copy()
#     result_worse = result_better.copy()
#     for i in range(len(makespan[0])):
#         for s1 in range(len(scheduler)):
#             for s2 in range(s1+1, len(scheduler)):
#                 if makespan[s1][i] < makespan[s2][i]:
#                     result_better[s1][s2] += 1
#                     result_worse[s2][s1] += 1
#                 elif makespan[s1][i] == makespan[s2][i]:
#                     result_equal[s1][s2] += 1
#                     result_equal[s2][s1] += 1
#                 else:
#                     result_better[s2][s1] += 1
#                     result_worse[s1][s2] += 1
#     result_better = np.round(100 * result_better / len(makespan[0]),2)
#     result_euqal = np.round(100 * result_equal / len(makespan[0]),2)
#     result_worse = np.round(100 * result_worse / len(makespan[0]),2)
#     draw_pairwise_comparison_table(scheduler, result_better, result_equal, result_worse)

        

