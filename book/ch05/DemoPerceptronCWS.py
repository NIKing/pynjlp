import sys
sys.path.append('/pynjlp')

from nlp.model.perceptron.CWSTrainer import CWSTrainer
from nlp.model.perceptron.PerceptronLexicalAnalyzer import PerceptronLexicalAnalyzer

from src.corpus.MSR import MSR

def train():
    model = CWSTrainer().train(MSR.TRAIN_PATH, '/pynjlp/data/test/msr_cws_preceptron.bin', maxIteration=2, threadNum=0).getModel()
    segment = PerceptronLexicalAnalyzer(model)

    return segment

def load_model():
    return PerceptronLexicalAnalyzer(cwsModelFile = '/pynjlp/data/test/msr_cws_preceptron.bin')
    
if __name__ == '__main__':
    segment = train()

    #segment = load_model()
    sentences = [
        "王思斌，男，１９４９年１０月生。",
        "山东桓台县起凤镇穆寨村妇女穆玲英",
        "现为中国艺术研究院中国文化研究所研究员。",
        "我们的父母重男轻女",
        "北京输气管道工程",
    ]
    
    for sent in sentences:
        res = segment.seg(sent)
        print(res)


