from types import SimpleNamespace

run_config = SimpleNamespace(**{
    # report
    "scheduler_show": ["SDTS","GenDoc", "HEFT","Optim"],

    # run_all
    # "scheduler_run": ["SDTS","GenDoc", "HEFT", "Optim"],
    "scheduler_run": ["Optim"],

    # generate_dataset
    "K": 5,
    "lfr": 2,
    "ccr": 0.5,
    "dcr": 5
})