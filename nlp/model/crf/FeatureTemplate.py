import re

class FeatureTemplate():
    pattern = re.compile("%x\[(-?\d*),(\d*)\]")

    def __init__(self):
        self.template = ""
        
        # 比如：U3:%x[-2,0]%x[-1,0]
        # 每个部分%x[-2,0]的位移，其中int[0]储存第一个数（-2），int[1]储存第二个数（0)
        self.offsetList = []
        self.delimiterList = []         # 分隔符, %x


    def load(self, byteArray):
        template = byteArray.next()
        size = byteArray.next()

        for i in range(size):
            self.offsetList.append([byteArray.next(), byteArray.next()])

        size = byteArray.next()
        for i in range(size):
            self.delimiterList.append(byteArray.next())

        return True

    @staticmethod
    def create(template):
        featureTemplate = FeatureTemplate()
        
        featureTemplate.delimiterList = []
        featureTemplate.offsetList = []
        featureTemplate.template = template

        matcher = FeatureTemplate.pattern.match(template)
        start = 0
        while matcher:
            featureTemplate.delimiterList.append(template[start:matcher.start()])
            featureTemplate.offsetList.append([int(matcher.group(1)), int(matcher.group(2))])

            start = matcher.end()

        return featureTemplate
