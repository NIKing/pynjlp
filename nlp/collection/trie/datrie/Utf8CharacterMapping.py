"""Utf8 到 int 映射"""
class Utf8CharacterMapping():
    serialVersionUID = '-6529481088518753872L'
    N = 256
    EMPTYLIST = [0]

    def getInitSize(self):
        return self.N

    def getCharsetSize(self):
        return self.N

    def toIdList(self, key):
        """
        转换字符为编号列表
        -param key 被转换的对象
        """

        # 处理key不同值的时候不同的转换过程
        if isinstance(key, str):
            return self.strToIdList(key)
        
        if isinstance(key, int):
            return self.intToIdList(key)

        if isinstance(key, list):
            return self.listToIdList(key)


    def strToIdList(self, key):
        """ 
        处理 字符 转换 intList
        -param key 等待转化的字符串
        return utf-8数字向量，每个值的值域在[0, 255]之间
        """

        # 将字符串编码为 UTF-8 字节数组
        key_bytes = key.encode('utf-8')
        
        #print('utf8=', key, key_bytes)
        # "你好" => b'\xe4\xbd\xa0\xe5\xa5\xbd' 长度等于 6
        
        # 0xFF 是 16 进制表示的数值，其二进制形式为 11111111，即 8 个二进制位全部为1
        # 通过使用 & 按位与操作符，可以保留一个整数的低 8 位，同时将高位清零
        # 以上，通过 & 0xFF 操作，就是为了确保结果的值域在 0 到 255 之间
        res = [0] * len(key_bytes)
        for i in range(len(key_bytes)):
            res[i] = key_bytes[i] & 0xFF

        if len(res) == 1 and res[0] == 0:
            return self.EMPTYLIST

        return res

    def intToIdList(self, codePoint):
        """处理 int 转换 intList"""
        count = 0
        if codePoint < 0x80:
            count = 1
        elif codePoint < 0x800:
            count = 2
        elif codePoint < 0x10000:
            count = 3
        elif codePoint < 0x200000:
            count = 4
        elif codePoint < 0x4000000:
            count = 5
        elif codePoint <= 0x7fffffff:
            count = 6
        else:
            return EMPTYLIST;


        r = [0] * count
        if count == 6:
            r[5] = chr(0x80 | (codePoint & 0x3f))
            codePoint = codePoint >> 6
            codePoint |= 0x4000000
        elif count == 5:
            r[4] = chr(0x80 | (codePoint & 0x3f))
            codePoint = codePoint >> 6
            codePoint |= 0x200000
        elif count == 4:
            r[3] = chr(0x80 | (codePoint & 0x3f))
            codePoint = codePoint >> 6
            codePoint |= 0x10000
        elif count == 3:
            r[2] = chr(0x80 | (codePoint & 0x3f))
            codePoint = codePoint >> 6
            codePoint |= 0x800
        elif count == 2:
            r[1] = chr(0x80 | (codePoint & 0x3f))
            codePoint = codePoint >> 6
            codePoint |= 0xc0
        elif count == 1:
            r[0] = chr(codePoint)
        
        return r


    def listToIdList(self, ids):
        """处理 list 转换 intList"""
        key_bytes = [0] * len(ids)
        for i in range(len(ids)):
            key_bytes[i] = bytes(ids[i])

        return key_bytes.decode('utf-8')


    def toString(self, ids) -> str:
        """int 转换 utf8字符串"""

