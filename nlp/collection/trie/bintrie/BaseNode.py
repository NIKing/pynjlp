from enum import Enum
from abc import ABC, abstractmethod

class Status(Enum):
    UNDEFINED = 0
    NOT_WORD = 1
    WORD_MIDDLE = 2
    WORD_END = 3

class BaseNode(ABC):
    def __init__(self, char = '', status = '', value = None):
        self.child  = []
        self.value  = value
        self.status = status
        self.c      = char
    
    def getChar(self):
        return self.c
    
    def transition(self, path):
        cur = self.getChild(path)
        #print(len(self.child), cur.getChar())
        if cur == None or cur.status == Status.UNDEFINED:
            return None

        return cur

    def getValue(self):
        return self.value

    def setValue(self, value):
        self.value = value
    
    def walk(self, entrySet):

        # 在java的代码中，这保存的方式本来是要申明 TrieEntry 的，在这里直接使用 dict 方式保存
        if self.status == Status.WORD_MIDDLE or self.status == Status.WORD_END:
            entrySet[self.c] = self.value
        
        if len(self.child) <= 0:
            return

        for node in self.child:
            if not node:
                continue

            node.walk(entrySet)

    @abstractmethod
    def getChild(self, c):
        return None
    
    def compareTo(self, c):
        """
        根据unicode字符集规则比较字符大小
        比如 
            '江'的Unicode码点是6c5f，通过 ord() 转整数是27743 
            '池'的Unicode码点是6c20，通过 ord() 转整数是25220
        """
        #print(f'comp, self_c={self.c}. c={c}')
        #print(f'comp_res = {self.c > c}')
        
        if not isinstance(c, str):
            c = c.getChar()

        if self.c > c:
            return 1

        if self.c < c:
            return -1

        return 0
