
class Nature:
    
    # 仅用于始##始，不会出现在分词结果中
    begin = 'begin'
    
    # 仅用于终##终，不会出现在分词结果中
    end = 'end'
    
    # 名词
    n = 'n'

    # 其他专名
    nz = 'nz'

    # 学术词汇
    g = 'g'
    
    idMap = None
    values = []

    def __init__(self, name):

        self.name = name
        
        ordinal = len(self.idMap.keys())
        self.idMap[name] = ordinal 

        self.extended = Nature(len(self.idMap))

        if self.values:
            self.extended = self.values

        self.extended[ordinal] = self
        self.values = self.extended
        
