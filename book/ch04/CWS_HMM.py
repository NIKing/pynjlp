import sys
import os
sys.path.append('/pynjlp')

from nlp.model.hmm.FirstOrderHiddenMarkovModel import FirstOrderHiddenMarkovModel
#from nlp.model.hmm.SecondOrderHiddenMarkovModel import SecondOrderHiddenMarkovModel
from nlp.model.hmm.HMMSegmenter import HMMSegmenter
from nlp.model.hmm.Vocabulary import Vocabulary

import nlp.corpus.io.IOUtil as IOUtil
from nlp.dictionary.other.CharTable import CharTable

from src.corpus.MSR import MSR
from nlp.seg.common.CWSEvaluator import CWSEvaluator

def train(corpus, model):
    segmenter = HMMSegmenter(model)
    segmenter.train(corpus)
    
    #model.unLog()
    print('--初始概率- start ---')
    print(len(model.start_probability))
    print(model.start_probability)
    print('--初始概率- end ---')
    print('')
   
    print('--转移概率- start ---')
    print(len(model.transition_probability))
    for row in model.transition_probability:
        print(row)
    print('--转移概率- end ---')
    print('')


    print('----发射概率 start----')
    emission_probability = model.emission_probability
    print(len(emission_probability))
    print(len(emission_probability[0]))
    print(emission_probability[0][:10])
    print('-----发射概率 end------')

    #print(f'词表大小：{segmenter.vocabulary.trie.getSize()}')
    new_child = []
    for child_node in segmenter.vocabulary.trie.child:
        if child_node.value != None:
            new_child.append(child_node)
    
    print(f'当前训练数据中，所用字符总数量：{len(new_child)}')
    #print([(node.c, node.value, node.status) for node in new_child[:20]])
    #print([(node.c, node.value, node.status) for node in segmenter.vocabulary.trie.child[:20]])

    #print(segmenter.segment('商品和服务'))

    return segmenter

def load_model():
    #vocabulary = IOUtil.readlinesTxt('/hanlp/pyhanlp/tests/book/ch04/model/vocabulary.txt')
    
    start_probability = IOUtil.readlinesTxt('/hanlp/pyhanlp/tests/book/ch04/model/start_probability.txt')
    transition_probability = IOUtil.readlinesTxt('/hanlp/pyhanlp/tests/book/ch04/model/transition_probability.txt')
    emission_probability = IOUtil.readlinesTxt('/hanlp/pyhanlp/tests/book/ch04/model/emission_probability.txt')
    
    pi = [float(p) for p in start_probability[0].split(' ')]

    A =  []
    for row in transition_probability:
        A.append([float(p) for p in row.split(' ')])

    B =  []
    for row in emission_probability:
        B.append([float(p) for p in row.split(' ')])

    print(pi)
    print(A)
    print(len(B))
    print(len(B[0]))
    print(B[0][:10])

    model = FirstOrderHiddenMarkovModel(pi, A, B)
    vocabulary = Vocabulary().from_pretrained('/hanlp/pyhanlp/tests/book/ch04/model/vocabulary.txt')

    segmenter = HMMSegmenter(model, vocabulary)

    return segmenter

def calculate_vocabulary_size():
    lineList = IOUtil.readlinesTxt(MSR.TRAIN_PATH)
    
    print(MSR.TRAIN_PATH)
    print(f'共有：{len(lineList)}条句子')
    
    str_line_list = []
    for line in lineList:
        str_line_list.append(line)

    str_text = ''.join(str_line_list)

    vocabulary_set = set()
    for c in str_text:
        c = CharTable.convert(c)
        vocabulary_set.add(c)

    print(len(vocabulary_set))

    IOUtil.writeTxtByList('/pynjlp/log/vocabulary.txt', list(vocabulary_set))


if __name__ == '__main__':

    #segment = train(MSR.TRAIN_PATH, FirstOrderHiddenMarkovModel())
    #segment = train(MSR.TRAIN_PATH, SecondOrderHiddenMarkovModel())
    segment = load_model()

    result = CWSEvaluator.evaluate(segment, MSR.OUTPUT_PATH, MSR.GOLD_PATH, MSR.TRAIN_WORDS)
    print(list(zip(['P', 'R', 'F1', 'OOV-R', 'IV-R'], list(result))))

    #calculate_vocabulary_size()


