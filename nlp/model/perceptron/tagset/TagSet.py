from nlp.model.perceptron.common.TaskType import TaskType

class TagSet:
    def __init__(self, _type):
        self.taskType = _type
        self.stringIdMap = {}
        
        self.idStringMap = []
        self.allTags = []

    def __len__(self):
        return len(self.stringIdMap)

    def __getitem__(self, i):
        return self.idStringMap[i]

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

    def size(self):
        return len(self.stringIdMap)

    def stringOf(self, id):
        return self.idStringMap[id]

    def idOf(self, string):
        id = self.stringIdMap.get(string)
        if id == None:
            id = -1

        return id

    def getAllTags(self):
        return self.allTags

    def save(self, out):
        """保存标记类别序号、标记大小和标记内容"""
        out.append(list(TaskType).index(self.taskType))
        out.append(self.size())

        for tag in self.idStringMap:
            out.append(tag)
    
    def load(self, byteArray):
        size = byteArray.next()
        
        for i in range(size):
            tag = byteArray.next()
            
            self.idStringMap.append(tag)
            self.stringIdMap[tag] = i

        self.lock()
    
    def bosId(self):
        return self.size()

    def lock(self):
        self.allTags = [0] * self.size()
        for i in range(self.size()):
            self.allTags[i] = i
