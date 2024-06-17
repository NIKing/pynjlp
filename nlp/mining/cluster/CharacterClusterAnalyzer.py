from nlp.mining.cluster.ClusterAnalyzer import ClusterAnalyzer
from nlp.mining.cluster.SparseVector import SparseVector
from nlp.mining.cluster.Document import Document

from collections import defaultdict

class CharacterClusterAnalyzer(ClusterAnalyzer):
    def __init__(self):
        super().__init__()

        self.words = []
        self.char_id_map = defaultdict(int)
        self.id_char_map = defaultdict(str)
    
    def addWords(self, words):
        self.words = words

        for i, word in enumerate(words):
            vector = self.toVector(word)
            d = Document(i, vector)
            
            keys, values = [], []
            for key, val in vector.entrySet():
                keys.append(key)
                values.append(self.id_char_map[key])

            #print(i, keys, values)
            self.documents[i] = d

        #print(self.char_id_map)

    def toVector(self, word):
        vector = SparseVector()
        for w in word:
            id = self.id(w)
            f  = vector.get(id)

            if f == 0.0:
                f = 1.0
                vector.put(id, f)
            else:
                vector.put(id, ++f)

        return vector


    def id(self, character):
        if character in self.char_id_map:
            return self.char_id_map[character] 
        
        id = len(self.char_id_map)

        self.char_id_map[character] = id
        self.id_char_map[id] = character

        return id

        
