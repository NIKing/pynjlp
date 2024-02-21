from abc import ABC, abstractmethod

from nlp.model.hmm.Vocabulary import Vocabulary
from nlp.corpus.io.IOUtil import loadInstance

"""模型训练器"""
class HMMTrainer(ABC):
    def __init__(self, model, vocabulary = None):
        """
        -param model 模型（一阶/二阶模型）
        -param vocabulary 词表
        """
        self.model = model

        if vocabulary == None:
            self.vocabulary = Vocabulary()
        else:
            self.vocabulary = vocabulary
    
    @abstractmethod
    def convertToSequence(self, sentence):
        pass
    
    @abstractmethod
    def getTagSet(self):
        pass

    def train(self, corpus):
        """
        模型训练，在这里的训练实际上是整理语料库
        第一步，给语料库（已经分好词的句子）打上{B,M,E,S} 标签
        第二步，把字符和标签转换成编码(注意，这里是以字符为单位，字符具有动态组词能力)
        """
        if not corpus:
            print('没有可加载的语料库')
        
        # 把句子转换为{B,M,E,S}标签序列
        sequenceList = []
        lineList = loadInstance(corpus)

        for line in lineList:
            sequenceList.append(self.convertToSequence(line))

        # 获取分词的标注集
        tagSet = self.getTagSet()
        
        #print(f'共有句子{len(sequenceList)}条')
        #print(sequenceList[0])

        # 根据标签序列进行转换样本数据，样本列表为双维数组
        # 第一层，字符序列在词表中映射的编号
        # 第二层，标签序列的编号
        sampleList = [] * len(sequenceList)
        for sequence in sequenceList:
            sample = [[0] * len(sequence) for i in range(2)]
            
            i = 0
            for os in sequence:
                sample[0][i] = self.vocabulary.idOf(os[0])
                assert sample[0][i] != -1

                sample[1][i] = tagSet.add(os[1])
                assert sample[1][i] != -1

                i += 1

            sampleList.append(sample)
        
        #print(sampleList[:1])
        
        # 真正的训练在这里
        self.model.train(sampleList)
        self.vocabulary.mutable = False

