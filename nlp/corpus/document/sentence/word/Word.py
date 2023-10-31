class Word:

    def __init__(self, value = None, label = None):
        self.value = value
        self.label = label

    def toString(self):
        if not self.label:
            return self.value

        return self.value + '/' + self.label

    def getValue(self):
        return self.value

    def setValue(self, value):
        self.value = value

    def getLabel(self):
        return self.label

    def setLabel(self, label):
        self.label = label

    def length():
        return len(self.vlaue)
