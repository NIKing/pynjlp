class crf_learn():
    
    # 特征最低频次
    freq = 1
   
    # 最大迭代次数
    maxitr = 10

    # 成本-系数
    cost = 1.0
    
    # 收敛阈值
    eta = 0.0001
    
    # 线程数 
    thread = 3
    
    # 收缩
    shrinking_size = 20
    
    # 训练算法
    algorithm = "CRF-L2"

    convert = False

    convert_to_text = False

    textmodel = False

    



