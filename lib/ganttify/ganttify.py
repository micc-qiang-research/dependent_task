import matplotlib.pyplot as plt
import numpy as np
from .legend import Legend
from .bar import Bar

class Ganttify:
    def __init__(self, json_obj):
        self.title = json_obj["title"]
        self.label = json_obj["label"]
        self.legend = []

        for legend in json_obj["legend"]:
            self.legend.append(Legend(legend))

        self.rows = json_obj["rows"]
        self.columns = json_obj["columns"]

        self.bars = []

        for bar in json_obj["bars"]:
            self.bars.append(Bar(bar))

        self.bar_height = json_obj["bar_height"]
        self.finish_time = json_obj["finish_time"]


    def run(self):

        fig = plt.figure()
        ax = plt.subplot(111)

        plt.title(self.title)
        plt.xlabel(self.label)


        plotted_bars = []
        legend_texts = []
        all_text = set()

        for bar in self.bars:
            plot_bar = ax.barh(self.get_row_index(bar), bar.width, left=bar.start, align='center', alpha=0.4, color=bar.color, height=self.bar_height)
            
            text = self.get_legend_text_by_color(bar.color)
            if text.startswith("task"):
                ax.text(bar.start + bar.width / 4, self.get_row_index(bar) - self.bar_height / 4, text)
            if text not in all_text:
                plotted_bars.append(plot_bar)
                legend_texts.append(text)
                all_text.add(text)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        ax.legend(plotted_bars, legend_texts, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.yticks(np.arange(len(self.rows)), self.rows)

        plt.axvline(self.finish_time, linestyle='--')
        plt.text(self.finish_time+0.1, -0.5, f"finish: {self.finish_time:.2f}", fontsize=10)

        plt.show()

    def get_row_index(self, bar):

        for i, row in enumerate(self.rows):
            if row == bar.row:
                return i

        return -1

    def get_legend_text_by_color(self, color):
        for leg in self.legend:
            if leg.color == color:
                return leg.text

        return "Error"
