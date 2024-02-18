
class TagSet:

    def __init__(self, _type):
        self.taskType = _type
        self.stringIdMap = {}
        
        self.idStringMap = []
        self.allTags = []


    def add(self, tag) -> int:
        """
        1，获取标签的编号
        2，添加标签和编号的映射关系
        """
        id = self.stringIdMap.get(tag, None)

        if id == None:
            id = len(self.stringIdMap)
            self.stringIdMap[tag] = id
            self.idStringMap.append(tag)

        return id
    
    def lock(self):
        self.allTags = [0] * self.size()

        for i in range(self.size()):
            self.allTags[i] = i

    def size(self):
        return len(self.stringIdMap)
