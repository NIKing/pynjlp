###
#模拟java的hashCode, python中的hash不够完美，明明是两个相近的字符，距离却相差万里。比如 "hash('池') - hash('江')" 
###
from pyhanlp import *

class GetHashCode:

    def convert_n_bytes(self, n, b):
        bits = b * 8
        return (n + 2 ** (bits - 1)) % 2 ** bits - 2 ** (bits - 1)

    def convert_4_bytes(self, n):
        return self.convert_n_bytes(n, 4)

    @classmethod
    def getHashCode(cls, s):
        h = 0
        n = len(s)
        for i, c in enumerate(s):
            h = h + ord(c) * 31 ** (n - 1 - i)
        return cls().convert_4_bytes(h)


def hash_code(s):
    return abs(GetHashCode.getHashCode(s))

def char_hash(c) -> int:
    return JClass('java.lang.Character')(c).hashCode()

#print(hash_code('池'))
#print(char_hash('池'))
#print(hash_code('池') - hash_code('江'))
#print(char_hash('池') - char_hash('江'))

