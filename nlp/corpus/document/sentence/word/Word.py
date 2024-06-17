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

    @staticmethod
    def create(param):
        """
        通过一个参数构建一个单词
        -param param 比如 人民网/nz
        return 一个单词
        """
        if not param:
            return None

        cutIndex = param.rfind('/')
        if cutIndex == -1 or cutIndex == len(param) - 1:
            print(f'使用参数{param}构建Word失败')
            return

        return Word(param[:cutIndex], param[(cutIndex + 1):])

