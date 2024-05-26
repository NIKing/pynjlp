from nlp.model.perceptron.common.TaskType import TaskType

from nlp.model.perceptron.tagset.TagSet import TagSet
from nlp.model.perceptron.tagset.CWSTagSet import CWSTagSet

class FeatureMap():
    def __init__(self, tagSet = None, mutable = False):
        self.size = 0
        self.tagSet  = tagSet
        self.mutable = mutable
    
    def allLabels(self) -> int:
        return self.tagSet.getAllTags()

    def bosTag(self):
        return self.tagSet.size()

    def save(self, out):
        """保存模型，已被ImmutableFeatureMDataMap类替换"""
        self.tagSet.save(out)
        out.append(self.size)

    def load(self, byteArray):
        """加载模型，已被ImmutableFeatureMDataMap类替换"""
        self.loadTagSet(byteArray)
        self.size = byteArray.nextInt()

    def loadTagSet(self, byteArray):
        type_byte = byteArray.next()
        taskType = list(TaskType)[type_byte].value

        taskTypeMap = {
            'CLASSIFICATION': TagSet(TaskType.CLASSIFICATION),
            'CWS': CWSTagSet()
        }
         
        # 加载分类标签对象，根据任务类型分配不同对象
        self.tagSet = taskTypeMap[taskType]
        self.tagSet.load(byteArray)

    
        


