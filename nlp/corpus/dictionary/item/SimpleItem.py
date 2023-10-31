
from collections import defaultdict

class SimpleItem:
    def __init__(self):
        self.labelMap = defaultdict(int)

    def addLabel(self, label):
        frequency = self.labelMap.get(label)
        if not frequency:
            frequency = 1
        else:
            frequency += 1

        self.labelMap[label] = frequency

    @staticmethod
    def create(self):
        pass
