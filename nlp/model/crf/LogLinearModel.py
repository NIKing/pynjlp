from nlp.model.perceptron.common.TaskType import TaskType
from nlp.model.perceptron.model.LinearModel import LinearModel
from nlp.model.perceptron.feature.MutableFeatureMap import MutableFeatureMap

from nlp.model.perceptron.tagset.TagSet import TagSet
from nlp.model.perceptron.tagset.CWSTagSet import CWSTagSet
from nlp.model.perceptron.tagset.NERTagSet import NERTagSet

from nlp.model.crf.FeatureTemplate import FeatureTemplate
from nlp.model.crf.FeatureFunction import FeatureFunction

from nlp.utility.Predefine import Predefine
from nlp.corpus.io.IOUtil import lineIterator

"""对数线性模型"""
class LogLinearModel(LinearModel):

    def __init__(self, featureMap = None, parameter = None, modelFile = ""):
        """
        -param modelFile str HanLP的.bin格式，或CRF++的.txt格式（将会自动转换为model.txt.bin，下次会直接加载.txt.bin）
        """
        super().__init__(featureMap, parameter)

        self.featureTemplateArray = []

        if modelFile == "":
            return
        
        # 加载bin
        if modelFile.endswith(Predefine.BIN_EXT):
            self.load(modelFile)
            return
        
        # 加载bin
        binPath = modelFile + Predefine.BIN_EXT
        try:
            self.load(binPath)
            return
        except Exception as e:
            print(f'模型加载失败：{e}')

        # 加载txt, 转换为bin
        self.convert(modelFile, binPath)


    def loadByteArray(self, byteArray):
        """重载父类方法"""
        if not super().loadByteArray(byteArray):
            return False

        size = byteArray.next()
        for i in range(size):
            featureTemplate = FeatureTemplate()
            featureTemplate.load(byteArray)
            
            self.featureTemplateArray.append(featureTemplate)
        
        return True

    def convert(self, txtFile, binFile):
        
        lineIter = lineIterator(txtFile)
        if not lineIter:
            print('空白文件')
            return

        print(next(lineIter))           # version
        print(next(lineIter))           # cost-factor

        maxid = int(next(lineIter)[len('maxid:'):])
        print(next(lineIter))           # xsize

        next(lineIter)  # blank
        
        # 读取标记
        tagSet = TagSet(TaskType.CLASSIFICATION)
        line = next(lineIter)
        while line != "":
            tagSet.add(line)
            line = next(lineIter)
        
        # 标记转换
        tagSet.type = self.guessModelType(tagSet)
        if tagSet.type == CWS:
            tagSet = CWSTagSet(tagSet.idOf('B'), tagSet.idOf('M'), tagSet.idOf('E'), tagSet.idOf('S'))
        elif tagSet.type == NER:
            tagSet = NERTagSet(tagSet.idOf('O'), tagSet.tags())

        tagSet.lock()
        
        self._set_featureMap(tagSet)
        self._set_parameter(tagSet, lineIter)

        print('文本读取完毕，开始转换模型数据到bin')
        self._to_bin()
            
    def _set_featureMap(self, tagSet):
        """特征空间"""
        self.featureMap = MutableFeatureMap(tagSet)
    
    def _set_parameter(self, tagSet, lineIter):
        """设置特征权重参数"""
        sizeOfTagSet = tagSet.size()
        
        # n元语法模型特征模版和转移特征模版
        featureTemplateList, matrix = [], None
        line = next(lineIter)
        while line != "":
            # 不是"B" 就是 n元语法模型
            if "B" != line:
                featureTemplate = FeatureTemplate.create(line)
                featureTemplateList.append(featureTemplate)
            else:
                matrix = [[sizeOfTagSet], [sizeOfTagSet]]

            line = next(lineIter)

        self.featureTemplateArray = featureTemplateList
        
        # 权重
        featureFunctionList = []    # 保存权重值
        b = -1      # 转换矩阵的权重位置
        if matrix != None:
            args = next(lineIter).split(" ", 2) # 0 B
            b = int(args[0])
            featureFunctionList.append((b, None))
        
        featureFunctionMap = {}
        line = next(lineIter)
        while line != "":
            args = line.split(" ", 2)
            charArray = args[1]
            featureFunction = FeatureFunction(charArray, sizeOfTagSet)
            
            featureFunctionMap[args[1]] = featureFunction
            featureFunctionList.append((int(args[0]), featureFunction))

            line = next(lineIter)

        for entry in featureFunctionList:
            fid, featureFunction = entry[0], entry[1]

            if fid == b:

                for i in range(sizeOfTagSet):
                    for j in range(sizeOfTagSet):
                        matrix[j][j] = float(next(lineIter))

            else:

                for i in range(sizeOfTagSet):
                    featureFunction.w[i] = float(next(lineIter))
        
        # 按理说到这里读区完毕了
        if lineIter:
            print('文本读取有残留，可能出问题', txtFile)
        
        # 注意这里开始进行 paramenter 的获取
        transitionFeatureOffset = (sizeOfTagSet + 1) * sizeOfTagSet
        self.parameter = [0.0] * (trainsitionFeatureOffset + len(featureFunctionMap) * sizeOfTagSet)

        if matrix != None:
            for i in range(sizeOfTagSet):
                for j in range(sizeOfTagSet):
                    self.parameter[i * sizeOfTagSet + j] = matrix[i][j]

        for entry in featureFunctionList:
            id, f = entry[0], entry[1]
            if f == None:
                continue

            feature = str(f.o)
            for tid in range(len(featureTemplateList)):
                template = featureTemplateList[tid]
                iterator = template.delimiterList
                
                header = iterator[0]
                if feature.statswith(header):
                    fid = self.featureMap.idOf(feature[:len(header)] + tid)
                    for i in range(sizeOfTagSet):
                        self.parameter[fid * sizeOfTagSet + i] = float(f.w[i])

                    break
    
    def _to_bin(self):
        # 保存特征映射值和特征权重
        out = []
        
        # 保存双数组
        self.featureMap.save(out)

        # 保存权重参数, 数组的位置代表特征映射，位置中的值表示权重
        for aParameter in self.parameter:
            out.append(aParameter)

        # 保存特征模版
        out.append(len(self.featureTemplateArray))
        for template in self.featureTemplateArray:
            template.save(out)
        
        
        writeListToBin(modelFile, out) 

    def guessModelType(self, tagSet):
        """推测模型类型"""

        if len(tagSet) == 4 and \
                tagSet.idOf('B') != -1 and \
                tagSet.idOf('M') != -1 and \
                tagSet.idOf('E') != -1 and \
                tagSet.idOf('S') != -1:
                    return TaskType.CWS
        
        # 具有复合词【B-, M-, E-, O】标签，命名实体识别
        if tagSet.idOf('O') != -1:
            for tag in tagSet:
                parts = tag.split('-')
                if len(parts) > 1:
                    if len(parts[0]) == 1 and "BMES" in parts[0]:
                        return TaskType.NER

        return TaskType.POS
    
    def getFeatureTemplateArray(self):
        return self.featureTemplateArray


