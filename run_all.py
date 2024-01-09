from dataloader import *
from scheduler import *
from util import *
from analysis import Analysis

import os
import pandas as pd
import pickle
import multiprocessing as mp
from glob import glob
import warnings
import logging
from util import get_in_result_path
from config import run_config

warnings.filterwarnings('ignore',category=RuntimeWarning)
logging.basicConfig(filename="Error.log", level=logging.DEBUG)

def solve(tuple_val):
    idx,filename = tuple_val
    print("Evaluating {}".format(idx))
    val = filename.split('/')[2].split('.json')[0].split('_')
    result = []
    param = dict(zip(keys, val))

    for _ in range(n_trials):
        try:
            data = DataByJson(filename) # read Data
            # scheduler = ["SDTS","GenDoc", "HEFT", "Optim"]
            scheduler = run_config.scheduler_run
            for s in scheduler:
                sched = eval(s)
                param['makespan_'+s] = Analysis(data, *sched(data, None).schedule()).summarize()
            result.append(param.copy())
        except:
            # logging.error("Error occured", exc_info=True)
            # msg = 'filename: {}, ccr: {}, b: {}, n_nodes: {}, p: {}\ncomp_matrix:\n{} adj_matrix:\n{}'.format(
            #     filename, param['ccr'], param['b'], inputs[0], inputs[1], inputs[2], inputs[3])
            # logging.info(msg)
            print("Error occure")
    
    return result

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
    print("Data saved!")


# n_trials = 5
# minalpha = 20
# maxalpha = 150
# n = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400]
# fat = [0.1, 0.4, 0.8]
# density = [0.2, 0.8]
# regularity = [0.2, 0.8]
# jump = [1,2,4]
# ccr = [0.1, 0.25, 0.5, 0.8, 1, 2, 5, 8, 10, 15, 20, 25, 30]
# b = [0.1, 0.2, 0.5, 0.75, 1, 2]
# p = [4,8,16,32]

n_trials = 1
batch_number = 4

keys = ['n', 'fat', 'density', 'regularity', 'jump']

filenames = glob('data/json/*.json')

cpu_cnt = 4

pool = mp.Pool(cpu_cnt)
print('Using {} cores'.format(cpu_cnt))

# columns = ['n', 'fat', 'density', 'regularity', 'jump', 'ccr','b','p', 'makespan_HEFT', 'makespan_PSO', 'makespan_IPEFT']
data = []
chunk_size = len(filenames)//batch_number
for i in range(batch_number):
    if i==batch_number-1:
        result_list = pool.map(solve, enumerate(filenames[i*chunk_size:]))
    else:
        result_list = pool.map(solve, enumerate(filenames[i*chunk_size:(i+1)*chunk_size]))
    data.extend([result for sublist in result_list for result in sublist])
save("data.pkl", data)