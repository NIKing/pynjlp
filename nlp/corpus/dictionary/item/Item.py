import sys
sys.path.append('/pnjulp')

from nlp.corpus.dictionary.item.SimpleItem import SimpleItem

class Item(SimpleItem):
    def __init__(self, key = '', label = ''):
        super().__init__()
        self.key = key
        self.labelMap[label] = 1
    
    @staticmethod
    def create(self, param):
        pass

    def toString(self):
        strLabel = " ".join([key +" "+ str(val) for key, val in self.labelMap.items()])
        return self.key + " " + strLabel
