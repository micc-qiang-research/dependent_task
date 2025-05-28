from types import SimpleNamespace

run_config = SimpleNamespace(**{

    # 0: 总测试，选择scheduler中所有调度器，然后调度
    # 1: 测试deploy策略的有效性，sequence策略固定
    # 2: 测试sequence策略的有效性，deploy策略固定
    "setting": 0, # 0: 总测试，1：deploy，2: sequence
    
    "default_deploy": "LPTS",
    # "default_deploy": "GenDoc",
    "default_sequence": "DALP",

    # 部署策略种类
    "deploy": ["HEFT", "GenDoc", "SDTS", "LCAA", "LPTS"],
    # "deploy": ["HEFT", "GenDoc", "SDTS", "LCAAP", "SDTSPlus"],

    # 序列化下载策略种类
    "sequence": ["FCFS", "LOPO", "CNTR", "GLSA", "DALP"],

    # 调度器：综合了部署和序列化下载决策
    "scheduler": ["HEFT", "GenDoc", "SDTS", "LASA", "Proposed"],
    # "scheduler": ["HEFT", "GenDoc", "SDTS", "LASA", "SDTSPlus"],
    
    "style": [['c','o'], ['g','s'], ['royalblue', 'v'], ['m','^'], ['r','D']],

    "linestyle" : [(0, (5, 1)),'dotted','dashed','dashdot','solid'],

    # 调度策略的 部署+序列化下载决策
    "HEFT": ["HEFT", "FCFS"],
    "GenDoc": ["GenDoc", "FCFS"],
    "SDTS": ["SDTS", "FCFS"],
    "LASA": ["LCAA","GLSA"],
    "SDTSPlus": ["SDTSPlus", "DALP"],
    "Proposed": ["LPTS", "DALP"],
    
    # generate_dataset
    "K": 5,
    "ccr": 1.0,
    "lfr": 4.0,
    "dcr": 4.0, # 平均有20个镜像块

    "range_K": [5],
    "range_ccr": [0.1, 0.5, 1.0, 1.5, 2.0],
    "range_lfr": [1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
    "range_dcr": [1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
    # "range_lfr": [1.0, 3.0, 5.0, 7.0, 10.0], # layer和func的数量关系
    # "range_dcr": [1.0, 2.0, 5.0, 8.0, 10.0], # 镜像的平均size

    # "range_K": [5],
    # "range_ccr": [1.0],
    # "range_lfr": [5.0], # layer和func的数量关系
    # "range_dcr": [5.0], # 镜像的平均size
})