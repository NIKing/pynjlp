from nlp.model.perceptron.tagset.TagSet import TagSet
from nlp.model.perceptron.common.TaskType import TaskType

class CWSTagSet(TagSet):
    
    def __init__(self, b = None, m = None, e = None, s = None):
        super().__init__(TaskType.CWS)
        
        if b == None or m == None or e == None or s == None:
            self.B = self.add('B')
            self.M = self.add('M')
            self.E = self.add('E')
            self.S = self.add('S')
        else:
            self.B = b
            self.M = m
            self.E = e
            self.S = s

            id2tag = ['B', 'M', 'E', 'S']

            for tag in id2tag:
                self.add(tag)

        self.lock()


