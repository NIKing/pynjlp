from nlp.model.perceptron.tagset.TagSet import TagSet
from nlp.model.perceptron.common.TaskType import TaskType

"""命名实体识别标签类"""
class NERTagSet(TagSet):
    def __init__(self, o, tags = None):
        super(TaskType.NER)
        
        self.O_TAG_CHAR = 'O'
        self.B_TAG_PREFIX = "B-"
        self.B_TAG_CHAR = 'B'
        self.M_TAG_PREFIX = "M-"
        self.E_TAG_PREFIX = "E-"
        self.S_TAG = "S"
        self.S_TAG_CHAR = 'S'

        # 非NER
        if o:
            self.O = o
        else:
            self.O = self.add(this.O_TAG)     

        if tags:
            for tag in tags:
                self.add(tag)

                label = NERTagSet.posOf(tag)
                if len(label) != len(tag):
                    this.nerLabels.append(label)
    
    @staticmethod
    def posOf(tag):
        index = tag.find('-')
        if index != -1:
            return tag

        return tag[index + 1:]
