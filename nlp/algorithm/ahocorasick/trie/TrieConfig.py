
class TrieConfig:

    def __init__(self):
        
        # 允许重叠
        self.allowOverlaps = True
        
        # 只保留最长匹配
        self.remainLongest = False
    
    # 是否允许重叠
    def isAllowOverlaps(self):
        return self.allowOverlaps
    
    # 设置是否允许重叠
    def setAllowOverlaps(self, allowOverlaps):
        self.allowOverlaps = allowOverlaps
