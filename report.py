import pandas as pd
import pickle
from util import prepare_draw_cdf, draw_cdf, get_in_result_path
import os

with open(get_in_result_path('data.pkl'), 'rb') as handle:
    data = pickle.load(handle)
    scheduler = ["SDTS","GenDoc", "HEFT", "Optim"]
    makespan = []
    for s in scheduler:
        makespan.append(list(data["makespan_"+s]))

    prepare_draw_cdf(makespan)
    draw_cdf(len(scheduler), scheduler, "lat_cdf.csv")
    

