from collections import defaultdict

# 为了让Dict支持 entrySet() 特意进行写了一个包装类
class DefaultDictEntrySet(defaultdict):
    def entrySet(self):
        return set(self.items())
