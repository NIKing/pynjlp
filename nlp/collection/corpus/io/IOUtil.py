
def loadDictionary(path):
    
    lineList = readlinesText(path)
    splitter, storage = "\t", {}

    for line in lineList:
        
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

def readlinesText(path):
    if not path:
        return []

    line_list = []
    try:
        with open(path, 'r', encoding='utf8') as f:
            line_list = f.readlines()
    except IOError as e:
        print(f'读取失败:{e}')

    return line_list

