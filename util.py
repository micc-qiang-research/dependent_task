
import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue

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
    "#888888", # prepare
    "#000000", # download
    "#469990", # user
]

prepare_color = colors[-3]
download_color = colors[-2]
user_color = colors[-1]

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
    weights = nx.get_edge_attributes(G, "weight")
    weights = {e: weights[e]["weight"] for e in weights}
    nx.draw_networkx(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.show()

def output_gantt_json(name, rows, bars, finish_time):

    legend = ""
    legend_str = "{{\"color\": \"{}\", \"text\": \"{}\"}},"
    for i, color in enumerate(colors[:-3]):
        legend = legend + legend_str.format(color, "task_{}".format(i+1))
    legend = legend + legend_str.format(prepare_color, "Prepare")
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

    with open("gantt.json", "w") as f:
        f.write(template.format(name, legend, rows, bars, finish_time))


def draw_gantt(file="gantt.json"):
    import json
    from lib.ganttify.ganttify import Ganttify

    json_obj = json.load(open(file))

    gantt = Ganttify(json_obj)
    gantt.run()
