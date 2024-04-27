import numpy as np
from nlp.corpus.io.IOUtil import readBinToList

class ByteArrayStream():

    def __init__(self, path):
        self.arr_bytes = readBinToList(path)
        self.offset = 0
    
    def next(self, count = 1):
        last_offset = self.offset
        self.offset += count
        
        res = self.arr_bytes[last_offset:self.offset]
        return res[0] if count == 1 else res

    def nextInt(self, count = 1):
        arr_bytes = self.arr_bytes[self.offset:]
        self.offset += (np.dtype(np.int32).itemsize * count)

        return np.frombuffer(arr_bytes, dtype=np.int32, count=count)

    def nextFloat(self):
        arr_bytes = self.arr_bytes[self.offset:]
        self.offset += (np.dtype(np.float64).itemsize * count)

        return np.frombuffer(arr_bytes, dtype=np.float64, count=count)
