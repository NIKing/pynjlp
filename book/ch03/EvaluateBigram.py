import sys
sys.path.append('/pynjlp')

from book.ch03.NgramSegment import train_bigram, load_bigram
from src.corpus.MSR import MSR
from nlp.seg.common.CWSEvaluator import CWSEvaluator 

if __name__ == '__main__':
    MSR_MODEL_PATH = MSR.MODEL_PATH + '_ngram'

    #train_bigram(MSR.TRAIN_PATH, MSR_MODEL_PATH)
    segment = load_bigram(MSR_MODEL_PATH, viterbi = True)
    #print(segment.seg("如今问起他的感觉，言语中不无得意：“以前怕手跟不上脑子灵感飞了，现在发愁的是脑袋转得太慢了。｀得心应手＇大概就是这个意思吧。”"))
    #print(segment.seg("再往远些看，随着汉字识别和语音识别技术的发展，中文计算机用户将跨越语言差异的鸿沟，在录入上走向中西文求同的道路"))
    #print(segment.seg("庞朴一听，算了吧，回家后试着把连线从一个接口往另一个接口一换，成了！"))
    
    result = CWSEvaluator.evaluate(segment, MSR.OUTPUT_PATH, MSR.GOLD_PATH, MSR.TRAIN_WORDS)
    print(list(zip(['P', 'R', 'F1', 'OOV-R', 'IV-R'], list(result))))
