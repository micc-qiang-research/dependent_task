
import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue
import pandas as pd
import numpy as np
import os

result_path = "./__result__/"

colors = [
    "#E6194B",
    "#3CB44B",
    "#FFE119",
    "#4364DB",
    "#F58231",
    "#911EB4",
    "#42D4F4",
    "#F032E6",
    "#BFEF45",
    "#FABED4",
    "#732000",
    "#DCBEFF",
    "#9A6324",
    "#FFFAC8",
    "#800000",
    "#AAFFC3",
    "#808000",
    "#FFD8B1",
    "#000075",
    "#A9A9A9",
    "#888888", # prepare
    "#000000", # download
    "#469990", # user
]

prepare_color = colors[-3]
download_color = colors[-2]
user_color = colors[-1]

# 防止中文乱码
import matplotlib.pyplot as plt
font_name = "simhei"
# font_name = "simsun"
plt.rcParams['font.family']= font_name # 指定字体，实际上相当于修改 matplotlibrc 文件　只不过这样做是暂时的　下次失效
plt.rcParams['axes.unicode_minus']=False # 正确显示负号，防止变成方框

# 1. 全局字体大小设置
plt.rcParams.update({
    'font.size': 24,              # 基础字体大小
    'axes.labelsize': 24,         # 坐标轴标签字体大小
    'axes.titlesize': 24,         # 标题字体大小
    'xtick.labelsize': 22,        # x轴刻度标签字体大小
    'ytick.labelsize': 22,        # y轴刻度标签字体大小
    'legend.fontsize': 24,        # 图例字体大小
    'figure.titlesize': 26        # 图表标题字体大小
})

class PQueue:
    def __init__(self):
        self.queue = PriorityQueue()
        self.label = set()

    def get(self):
        return self.queue.get()        

    def put(self, item):
        if item[1] in self.label:
            return    
        self.label.add(item[1])
        return self.queue.put(item)

    def empty(self):
        return self.queue.empty()

def draw_dag(G):
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    # pos = nx.nx_agraph.graphviz_layout(G)
    weights = nx.get_edge_attributes(G, "weight")
    # weights = {e: weights[e]["weight"] for e in weights}
    nx.draw_networkx(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.show()

def output_gantt_json(name, rows, bars, finish_time):

    legend = ""
    legend_str = "{{\"color\": \"{}\", \"text\": \"{}\"}},"
    for i, color in enumerate(colors[:-3]):
        legend = legend + legend_str.format(color, "task_{}".format(i+1))
    # legend = legend + legend_str.format(prepare_color, "Prepare")
    legend = legend + legend_str.format(download_color, "Download")
    legend = legend + legend_str.format(user_color, "User")[:-1]

    json_rows = ["user"]
    json_rows.extend(rows)
    rows = ""
    for c in json_rows:
        rows = rows + "\"{}\",".format(c)
    rows = rows[:-1]

    template = """
{{
    "title": "{}",
    "label": "Time",
    "legend": [
        {}
    ],
    "rows": [
            {}
        ],
    "columns": [],
    "bars": [
        {}
    ],
    "bar_height": 0.5,
    "finish_time": {}
}}
    """

    with open(os.path.join(result_path,"gantt.json"), "w") as f:
        f.write(template.format(name, legend, rows, bars, finish_time))


def draw_gantt(file="gantt.json"):
    import json
    from lib.ganttify.ganttify import Ganttify

    json_obj = json.load(open(os.path.join(result_path,file)))

    gantt = Ganttify(json_obj)
    gantt.run()


def np_to_csv(data, path):
    pd.DataFrame(data).to_csv(path, index=None, header=None)

def prepare_draw_cdf(data):
    assert len(data) >= 2, "unmatched data length"
    ds = []
    prob = []
    inc = 100. / len(data[0])
    k = inc
    for i in range(len(data[0])):
        prob.append(k)
        k += inc
    ds.append(prob)
    for i in data:
        ds.append(sorted(i))
    ds = np.array(ds).T
    np_to_csv(ds, os.path.join(result_path, "lat_cdf.csv"))

# draw_cdf(2, ["RS", "RS+UPL"], "lat_cdf.csv")

def draw_cdf(n, label, filename = "lat_cdf.csv"):
    input_file = os.path.join(result_path,filename)
    basename, _ = os.path.splitext(filename)
    output_file = os.path.join(result_path,basename+".pdf")
    # plt.rcParams.update({"font.size":14})
    from config import run_config
    style = run_config.style
    plt.figure(figsize=(8, 6))

    #读取CSV文件
    data = pd.read_csv(input_file)

    prob = data.iloc[:,0]
    for i in range(n):
        delays = data.iloc[:,i+1]
        plt.plot(delays, prob, label=label[i], color=style[i][0], linewidth=3)

    #设置图表属性
    plt.axvline(x=50, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Makespan')
    plt.ylabel('比例（%）')
    plt.grid(True, linestyle='--', alpha=0.7) 
    # plt.xlim(0,80000)
    plt.legend()
    # plt.show()
    plt.savefig(output_file,bbox_inches='tight')
    plt.close()

def draw_cdf_ax(ax, n, label, filename = "lat_cdf.csv"):
    import pandas as pd
    from config import run_config
    style = run_config.style 
    input_file = os.path.join(result_path,filename)
    basename, _ = os.path.splitext(filename)

    #读取CSV文件
    data = pd.read_csv(input_file)
    max_delay = -1

    prob = data.iloc[:,0]
    for i in range(n):
        delays = data.iloc[:,i+1]
        ax.plot(delays, prob, label=label[i], color=style[i][0])
        max_delay = max(max_delay, max(delays))
        df = pd.DataFrame(np.array([delays,prob]).T, columns=['delay', 'prob'])
        df.to_csv(f'{result_path}/{label[i]}.txt')

    # ax.axvline(x=50, color='black', linestyle='--', linewidth=1)
    # ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(np.arange(0, max_delay+1, 25))
    #设置图表属性
    ax.set_xlabel('Makespan')
    ax.set_ylabel('Percent (%)') 
    # plt.xlim(0,80000)
    ax.legend()

def get_in_result_path(filename):
    return os.path.join(result_path, filename)


# 画结果比较图
from tabulate import tabulate
def draw_pairwise_comparison_table(scheduler, result_better, result_equal, result_worse):
    row = ['', '']
    row.extend(scheduler)
    table = [row]
    ss = 'better\nequal\nworst'
    cell = "{}%\n{}%\n{}%"
    # ['SDTS',  ss, 23,12,23,11]
    for i,sched in enumerate(scheduler):
        row = [sched, ss]
        for j,sched2 in enumerate(scheduler):
            better = result_better[i][j]
            equal = result_equal[i][j]
            worse = result_worse[i][j]
            if i == j:
                row.append("*")
            else:
                row.append(cell.format(better, equal, worse))
        table.append(row)
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

def calcImprove(arr :list):
    j = len(arr)-1
    for i in range(j):
        print((arr[i]-arr[j])/arr[i]*100)
