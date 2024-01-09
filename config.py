from types import SimpleNamespace

run_config = SimpleNamespace(**{
    # report
    "scheduler_show": ["SDTS","GenDoc", "HEFT"],

    # run_all
    "scheduler_run": ["SDTS","GenDoc", "HEFT"],

    # generate_dataset
    "K": 5,
    "lfr": 10,
    "ccr": 0.5,
    "dcr": 5
})