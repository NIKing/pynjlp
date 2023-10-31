import os

def loadDictionary(path):
    if not path:
        return []

    splitter, line_list = "\t", []
    try:
        with open(path, 'r', encoding='utf8') as f:
            line_list = f.readlines()
    except IOError as e:
        print(f'读取失败:{e}')
    
    storage = {}
    for line in line_list:
        
        param = line.rstrip('\n').split(splitter)
        
        # 每个关键词存在多组属性（以词性和词频为组），计算关键词存在特征总数
        natureCount = int((len(param) - 1) / 2)
        key = str(param[0])
        if natureCount == 0:
            storage[key] = None
            continue
        
        nature, frequency, totalFrequency = [], [], 0
        for i in range(natureCount):
            nature.append(param[(1 + 2 * i)])
            frequency.append(param[(2 + 2 * i)])
            totalFrequency += int(frequency[i])
        
        storage[key] = { 'nature': nature, 'frequency': frequency, 'totalFrequency': totalFrequency }
    
    return storage

def readlinesTxt(filePath):
    if not filePath:
        return
    
    line_list = []
    try:
        with open(filePath, 'r', encoding='utf8') as file:
            line_list = file.readlines()
    except Exception as e:
        print('读取失败', e)
    
    #return line_list
    return [w.rstrip('\n') for w in line_list]

def writeTxtByList(filePath, dataList): 
    if not filePath or len(dataList) <= 0:
        return

    try:
        with open(filePath, "w", encoding='utf8') as file:
           file.writelines('\n'.join(dataList))
    except IOError:
        print('写入失败', IOError)
