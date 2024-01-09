import pandas as pd
import pickle
from util import prepare_draw_cdf, draw_cdf, get_in_result_path, draw_pairwise_comparison_table
import numpy as np
from config import run_config

with open(get_in_result_path('data.pkl'), 'rb') as handle:
    data = pickle.load(handle)
    # scheduler = ["SDTS","GenDoc", "HEFT", "Optim"]
    scheduler = run_config.scheduler_show
    makespan = []
    for s in scheduler:
        makespan.append(list(data["makespan_"+s]))

    prepare_draw_cdf(makespan)
    draw_cdf(len(scheduler), scheduler, "lat_cdf.csv")

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
    result_euqal = np.round(100 * result_equal / len(makespan[0]),2)
    result_worse = np.round(100 * result_worse / len(makespan[0]),2)
    draw_pairwise_comparison_table(scheduler, result_better, result_equal, result_worse)

        

