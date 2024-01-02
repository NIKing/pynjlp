
class AtomNode:
    """原子分词节点"""

    sWord = ""
    nPOS  = -1

    def __init__(self, sWord, nPOS):
        self.sWord = sWord
        self.nPOS  = nPOS

