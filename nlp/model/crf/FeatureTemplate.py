import re

class FeatureTemplate():
    pattern = re.compile(r'%x\[(-?\d*),(\d*)\]', re.I)
            
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
    
    def save(self, out):
        out.append(self.template)

        out.append(len(self.offsetList))
        for offset in self.offsetList:
            out.append(offset[0])
            out.append(offset[1])
        
        out.append(len(self.delimiterList))
        for s in self.delimiterList:
            out.append(s)

    @staticmethod
    def create(template):
        featureTemplate = FeatureTemplate()
        
        featureTemplate.delimiterList = []
        featureTemplate.offsetList = []
        featureTemplate.template = template

        matchers = FeatureTemplate.pattern.finditer(template)
        #print(f'----{template}--{matchers}')

        start = 0
        for matcher in matchers:
            # start() 是获取正则表达式匹配的结果在源字符的起始位置
            # group(0) 返回匹配成功的整个子串，比如：%x[-1,0]
            # group(1) 返回匹配成功的整个子串第一组值，group(2)返回第二组值，不存在group(3)
            featureTemplate.delimiterList.append(template[start:matcher.start()])
            featureTemplate.offsetList.append([int(matcher.group(1)), int(matcher.group(2))])

            start = matcher.end()

        return featureTemplate
