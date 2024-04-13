class FeatureSortItem:
    def __init__(self, entry = None, parameter = None, tagSetSize = 0):
        if entry == None or parameter == None:
            return

        self.key    = entry.getKey()
        self.id     = entry.getValue()

        self.total  = 0

        for i in range(tagSetSize):
            self.total += abs(parameter[self.id * tagSetSize + i])
