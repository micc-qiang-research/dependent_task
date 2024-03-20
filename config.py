from types import SimpleNamespace

run_config = SimpleNamespace(**{
    # report
    # "scheduler_show": ["SDTS","GenDoc", "HEFT","Optim"],
    "scheduler_show": ["SDTS","GenDoc", "HEFT","LCAAP", "SDTSPlus"],

    # run_all
    # "scheduler_run": ["SDTS","GenDoc", "HEFT", "Optim"],
    # "scheduler_run": ["SDTS","GenDoc", "HEFT","LCAAP", "SDTSPlus"],
    "scheduler_run": ["SDTSPlus"],
    # "scheduler_run": ["SDTS","GenDoc", "HEFT", "LCAAP"],
    
    # generate_dataset
    "K": 5,
    "lfr": 2,
    "ccr": 0.5,
    "dcr": 5
})