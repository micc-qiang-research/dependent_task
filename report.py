import pandas as pd
import pickle
from util import prepare_draw_cdf, draw_cdf, get_in_result_path, draw_pairwise_comparison_table, draw_cdf_ax
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

# data_dir = "__result__/app30-new/"
# data_dir = "__result__/app30-new-部署/"
data_dir = "__result__/app30-new-序列化/"
# data_dir = "__result__/app50/"


class ExtractFileHelper:
    def get_all_files():
        return glob(data_dir + "data_*.pkl")
    
    def get_file(k, ccr, lfr, dcr):
        return data_dir + "data_{}_{}_{}_{}.pkl".format(k, ccr, lfr, dcr)

    def get_filename_by_ccr(ccr):
        return data_dir + "data_{}_{}_{}_{}.pkl".format(default_k, ccr, default_lfr, default_dcr)

    def get_filename_by_k(k):
        return data_dir + "data_{}_{}_{}_{}.pkl".format(k, default_ccr, default_lfr, default_dcr)

    def get_filename_by_lfr(lfr):
        return data_dir + "data_{}_{}_{}_{}.pkl".format(default_k, default_ccr, lfr, default_dcr)

    def get_filename_by_dcr(dcr):
        return data_dir + "data_{}_{}_{}_{}.pkl".format(default_k, default_ccr, default_lfr, dcr)

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

        pattern_k = data_dir + r'data_(.*?)_{}_.*'.format(default_ccr)
        pattern_ccr = data_dir + r'data_{}_(.*?)_.*'.format(default_k)
        pattern_lfr = data_dir + r'data_{}_{}_(.*?)_.*'.format(default_k, default_ccr)
        pattern_dcr = data_dir + r'data_{}_{}_{}_(.*?).pkl'.format(default_k, default_ccr, default_lfr)
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

def get_scheduler():
    res = []
    # scheduler
    if run_config.setting == 0:
        scheduler = run_config.scheduler
        for sched in scheduler:
            s = getattr(run_config, sched)
            res.append(f"{sched}")
    # deploy
    elif run_config.setting == 1:
        deploy = run_config.deploy
        default_sequence = run_config.default_sequence
        for d in deploy:
            res.append(f"{d}-{default_sequence}")
    else:
        sequence = run_config.sequence
        default_deploy = run_config.default_deploy
        for s in sequence:
            res.append(f"{default_deploy}-{s}")
    return res

scheduler = get_scheduler()

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
    vals = ExtractFileHelper.get_index(ltype, filenames) # 获取所有取值
    vals.sort()
    makespans = {s:[] for s in scheduler}
    for val in vals:
        file = get_filename[ltype](val) # 获取当前取值对应的文件
        with open(file, 'rb') as handle:
            data = pickle.load(handle)
            for s in scheduler:
                column = s
                makespans[s].append(data[column].mean().round(2))
                # makespans[s].append(data[column].median().round(2))
    print("--------------")
    print(typename[ltype])
    for s in scheduler:
        for i in range(len(vals)):
            print(s, vals[i], makespans[s][i])
    
    return makespans, vals

def draw_linear(x_name, x, ys):
    plt.rcParams.update({"font.size":14})
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

def draw_linear_ax(ax, x_name, x, ys):
    # print(ys["SDTSPlus"])
    marker = run_config.style
    # marker = ['o', 's', 'D', '^', 'v']
    for i,s in enumerate(scheduler):
        ax.plot(x, ys[s], label=s, color=marker[i][0], marker=marker[i][1])

    # plt.plot(x, [1,2,3,4,5])
    # 添加标题和轴标签
    # ax.set_title(x_name)
    ax.set_xlabel(x_name)
    ax.set_ylabel('Avg. Makespan')
    ax.set_xticks(x)

    # 添加图例
    ax.legend(fontsize=8)

def draw_sensibility(ltype, ax = None):
    makespans, vals = get_makespan(ltype)
    # print(makespans)
    # print(vals)
    if ax is None:
        draw_linear(typename[ltype], vals, makespans)
    else:
        draw_linear_ax(ax, typename[ltype], vals, makespans)

def get_makespan_total(filenames = None):
    if filenames is None:
        filenames = glob(data_dir + "data_*.pkl")
        # filenames = glob(data_dir + "data_*_*_*_1.0.pkl")
    # filenames = glob(data_dir + "data_*.pkl")
    # filenames = get_all_files[Type.CCR]()
    # filenames = get_all_files[Type.LFR]()
    # filenames = get_all_files[Type.DCR]()
    # filenames = [ExtractFileHelper.get_file(default_k, default_ccr, default_lfr, default_dcr)]

    makespan = [[] for _ in scheduler]
    for filename in filenames:
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
            # scheduler = ["HEFT", "GenDoc", "SDTS", "LASA", "Propose"]
            for i,s in enumerate(scheduler):
                makespan[i].extend(list(data[s]))
    
    # 计算平均Makespan减少的百分比
    mean_makespan = np.array(makespan).mean(axis=1)
    for i in range(len(scheduler)):
        print(scheduler[i], mean_makespan[i])
    idx = 4
    print("base scheduler: ", scheduler[idx])
    imporve_makespan = mean_makespan[idx]
    # print("improve", imporve_makespan)
    print("-------------")
    for i,s in enumerate(scheduler):
        if i==idx: continue
        print("%s %f %.2f%%" % (s, mean_makespan[i],(mean_makespan[i]-imporve_makespan)/mean_makespan[i]*100))
    print("--------------")
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

# 画四宫图
def draw_full():
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(2, 2, figsize=(6.5, 6))
    # plt.rcParams.update({"font.size":8})
    #设置主标题
    # f.suptitle('My Figure')
    #设置子标题
    ax[0][0].set_title('(a) CDF distribution')
    makespan = get_makespan_total()
    prepare_draw_cdf(makespan)
    draw_cdf_ax(ax[0][0], len(scheduler), scheduler, "lat_cdf.csv")

    ax[0][1].set_title('(b) CCR')
    draw_sensibility(Type.CCR, ax[0][1])

    ax[1][0].set_title('(c) LFR')
    draw_sensibility(Type.LFR, ax[1][0])

    ax[1][1].set_title('(d) DCR')
    draw_sensibility(Type.DCR, ax[1][1])

    # f.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.4, hspace=0.4)
    plt.show()

# 画二宫图
def draw_half():
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, 2, figsize=(6, 2.75))
    #设置主标题
    # f.suptitle('My Figure')
    #设置子标题
    ax[0].set_title('(a) DCR = 1')
    makespan = get_makespan_total(filenames = glob(data_dir + "data_*_*_*_1.0.pkl"))
    prepare_draw_cdf(makespan)
    draw_cdf_ax(ax[0], len(scheduler), scheduler, "lat_cdf.csv")

    ax[1].set_title('(b) DCR = 10')
    makespan = get_makespan_total(filenames = glob(data_dir + "data_*_*_*_10.0.pkl"))
    prepare_draw_cdf(makespan)
    draw_cdf_ax(ax[1], len(scheduler), scheduler, "lat_cdf.csv")

    f.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()

# draw_full()
draw_half()



        

