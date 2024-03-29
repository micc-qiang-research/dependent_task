from types import SimpleNamespace

run_config = SimpleNamespace(**{
    # report
    # "scheduler_show": ["SDTS","GenDoc", "HEFT","Optim"],
    "scheduler_show": ["SDTS","GenDoc", "HEFT","LCAAP", "SDTSPlus"],

    # run_all
    # "scheduler_run": ["SDTS","GenDoc", "HEFT", "Optim"],
    "scheduler_run": ["SDTS","GenDoc", "HEFT","LCAAP", "SDTSPlus"],
    # "scheduler_run": ["SDTSPlus"],
    # "scheduler_run": ["SDTS","GenDoc", "HEFT", "LCAAP"],
    
    # generate_dataset
    "K": 5,
    "ccr": 1.0,
    "lfr": 5.0,
    "dcr": 5.0, # 平均有25个镜像块

    "range_K": [5],
    "range_ccr": [0.1, 0.5, 1.0, 1.5, 2.0],
    "range_lfr": [1.0, 3.0, 5.0, 7.0, 10.0], # layer和func的数量关系
    "range_dcr": [1.0, 2.0, 5.0, 8.0, 10.0], # 镜像的平均size

    # "range_K": [5],
    # "range_ccr": [1.0],
    # "range_lfr": [5.0], # layer和func的数量关系
    # "range_dcr": [5.0], # 镜像的平均size
})