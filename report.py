import pandas as pd
import pickle
from util import prepare_draw_cdf, draw_cdf, get_in_result_path, draw_pairwise_comparison_table
import numpy as np
from config import run_config
from glob import glob
import re
import matplotlib.pyplot as plt
import argparse

'''
文件系统的格式为: data_k_ccr_lfr_dcr.pkl
若是对ccr做敏感性分析, 则保持其他参数不变, 提取ccr变量
'''

class Type:
    K = 0
    CCR = 1
    LFR = 2
    DCR = 3

typename = ["k", "ccr", "lfr", "dcr"]

default_k = run_config.K
default_ccr = run_config.ccr
default_lfr = run_config.lfr
default_dcr = run_config.dcr

class ExtractFileHelper:
    def get_all_files():
        return glob("__result__/data_*.pkl")
    
    def get_file(k, ccr, lfr, dcr):
        return "__result__/data_{}_{}_{}_{}.pkl".format(k, ccr, lfr, dcr)

    def get_filename_by_ccr(ccr):
        return "__result__/data_{}_{}_{}_{}.pkl".format(default_k, ccr, default_lfr, default_dcr)

    def get_filename_by_k(k):
        return "__result__/data_{}_{}_{}_{}.pkl".format(k, default_ccr, default_lfr, default_dcr)

    def get_filename_by_lfr(lfr):
        return "__result__/data_{}_{}_{}_{}.pkl".format(default_k, default_ccr, lfr, default_dcr)

    def get_filename_by_dcr(dcr):
        return "__result__/data_{}_{}_{}_{}.pkl".format(default_k, default_ccr, default_lfr, dcr)

    # 获取所有ccr文件
    def get_all_files_ccr():
        filenames = glob(ExtractFileHelper.get_filename_by_ccr("*"))
        return filenames

    def get_all_files_k():
        filenames = glob(ExtractFileHelper.get_filename_by_k("*"))
        return filenames

    def get_all_files_dcr():
        filenames = glob(ExtractFileHelper.get_filename_by_dcr("*"))
        return filenames

    def get_all_files_lfr():
        filenames = glob(ExtractFileHelper.get_filename_by_lfr("*"))
        return filenames
    
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


# 获取所有文件
get_all_files = [
    ExtractFileHelper.get_all_files_k,\
    ExtractFileHelper.get_all_files_ccr,\
    ExtractFileHelper.get_all_files_lfr,\
    ExtractFileHelper.get_all_files_dcr
]

get_filename = [
    ExtractFileHelper.get_filename_by_k, \
    ExtractFileHelper.get_filename_by_ccr, \
    ExtractFileHelper.get_filename_by_lfr, \
    ExtractFileHelper.get_filename_by_dcr
]

scheduler = run_config.scheduler_show

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
def get_makespan(ltype):
    filenames = get_all_files[ltype]()
    vals = ExtractFileHelper.get_index(ltype, filenames)
    vals.sort()
    makespans = {s:[] for s in scheduler}
    for val in vals:
        file = get_filename[ltype](val)
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
            for s in scheduler:
                column = "makespan_"+s
                makespans[s].append(data[column].mean().round(2))
    return makespans, vals

def draw_linear(x_name, x, ys):
    # print(ys["SDTSPlus"])
    for s in scheduler:
        plt.plot(x, ys[s], label=s)

    # plt.plot(x, [1,2,3,4,5])
    # 添加标题和轴标签
    plt.title(x_name)
    plt.xlabel(x_name)
    plt.ylabel('Avg. Makespan')
    plt.xticks(x)

    # 添加图例
    plt.legend()

    # 展示图形
    plt.show()

def draw_sensibility(ltype):
    makespans, vals = get_makespan(ltype)
    draw_linear(typename[ltype], vals, makespans)


def get_makespan_total():
    # filenames = glob("__result__/data_*.pkl")
    # filenames = get_all_files[Type.CCR]()
    # filenames = get_all_files[Type.LFR]()
    # filenames = get_all_files[Type.DCR]()
    filenames = [ExtractFileHelper.get_file(default_k, default_ccr, 5.0, 2.0)]

    makespan = [[] for _ in scheduler]
    for filename in filenames:
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
            # scheduler = ["SDTS","GenDoc", "HEFT", "Optim"]
            for i,s in enumerate(scheduler):
                makespan[i].extend(list(data["makespan_"+s]))
    return makespan

def prepare_pairwise_data(makespan):
    result_better = np.zeros((len(scheduler), len(scheduler)))
    result_equal = result_better.copy()
    result_worse = result_better.copy()
    for i in range(len(makespan[0])):
        for s1 in range(len(scheduler)):
            for s2 in range(s1+1, len(scheduler)):
                if makespan[s1][i] < makespan[s2][i]:
                    result_better[s1][s2] += 1
                    result_worse[s2][s1] += 1
                elif makespan[s1][i] == makespan[s2][i]:
                    result_equal[s1][s2] += 1
                    result_equal[s2][s1] += 1
                else:
                    result_better[s2][s1] += 1
                    result_worse[s1][s2] += 1
    result_better = np.round(100 * result_better / len(makespan[0]),2)
    result_equal = np.round(100 * result_equal / len(makespan[0]),2)
    result_worse = np.round(100 * result_worse / len(makespan[0]),2)
    return result_better, result_equal, result_worse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--cdf", action="store_true", help="show cdf")
    parser.add_argument("-s", "--sensitive", action="store_true", help="show sensitive")
    parser.add_argument("-p", "--pair", action="store_true", help="show pairwise table")
    args = parser.parse_args()
    return args

args = parse()
if args.cdf or args.pair:
    makespan = get_makespan_total()
    if args.cdf:
        prepare_draw_cdf(makespan)
        draw_cdf(len(scheduler), scheduler, "lat_cdf.csv")
    if args.pair:
        result_better, result_equal, result_worse = prepare_pairwise_data(makespan)
        draw_pairwise_comparison_table(scheduler, result_better, result_equal, result_worse)
if(args.sensitive):
    draw_sensibility(Type.CCR)
    draw_sensibility(Type.LFR)
    draw_sensibility(Type.DCR)




        

