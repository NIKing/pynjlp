"""键值对存储方式"""
class KeyValuePair:
    def __init__(self, dat):
        self.path   = []
        
        self.index  = 0 
        self.value  = -1
        self.key    = None
        self.currentBase = 0
        
        self.dat = dat
        self.path = [1]

        self.collection()
    
    def __inter__(self, index):
        return self.next()

    def size(self):
        return self.dat.getSize()

    def collection(self):
        _from = 1       # 根节点为1
        b = self.dat.base[_from]

        if self.dat.size <= 0:
            return
        
        while True:
            i = 0

            # 字符集大小 256, 为什么是 256?，最长字符长度，并不是循环 256次，在下面会重置为 0 ，重新开始找
            # 另外一个原因，在创建双数组字典树的时候，每个词语中所有字符都用[0,255]数字表示, 因此循环256可以找到任何一个字符
            while i < self.dat.charMap.getCharsetSize():
                # c != _from 代表节点转移不成功，跳过
                c = self.dat.check[b + i]
                if c != _from:
                    i += 1
                    continue
                
                # 在path中保存，base值和check值
                _from = b + i
                self.path.append(i)
                self.path.append(_from)
                
                b = self.dat.base[_from]
                i = 0                       # 从头开始找
                
                # 找到了终止符，且转移成功
                if self.dat.getCheck(b + self.dat.UNUSED_CHAR_VALUE) == _from:
                     
                    # 获取存在 path 数组中check值
                    ids = []
                    for j in range(1, len(self.path), 2):
                        ids.append(self.path[j])
                    
                    # 将 utf8 字符集转化成字符串, 并还原最后字符对应的值
                    self.value = self.dat.getLeafValue(self.dat.getBase(b + self.dat.UNUSED_CHAR_VALUE))
                    self.key   = self.dat.charMap.toString(ids)
                    
                    self.path.append(self.dat.UNUSED_CHAR_VALUE)
                    self.currentBase = b

                    return

    def key(self):
        return self.key

    def value(self):
        return self.value

    def getKey(self):
        return self.key

    def getValue(self):
        return self.value

    def setValue(self, v):
        value = self.bat.getLeafValue(v)
        self.setBase(self.currentBase + self.bat.UNUSED_CHAR_VALUE)

        self.value = v
        return v

    def hasNext(self):
        return self.index < self.dat.size

    def next(self):
        if self.index >= self.dat.size:
            raise ValueError('没有更多元素了')
        
        # index 记录迭代次数
        if self.index != 0:
            print('path', self.path, self.index)
            while len(self.path) > 0:
                charPoint = self.path.pop()
                base = self.path[-1]
                n = self.getNext(base, charPoint)
                print('n', n)
                print(f'{base} 到 {charPoint}转移失败')
                print('')
                
                if n != -1:
                    break
                
                # 只所以再pop一次，因为path是base值和check值组成一对。在while循环中，第一次pop应该获取check值
                self.path.pop()

        self.index += 1

        return self

    def getNext(self, parent, charPoint):
        """
        遍历下一个终止路径
        -param parent 父节点
        -param charPoint 子节点char
        """
        startChar  = charPoint + 1
        baseParent = self.dat.base[parent]
        from_ = parent

        print(f'parent={parent}', f'charPoint={charPoint}')
        print(f'baseParent={baseParent}', f'startChar={startChar}')

        for i in range(startChar, self.dat.charMap.getCharsetSize()):
            to_ = baseParent + i
            
            print('train--', f'i={i}', f'baseParent={baseParent}',  f'p={to_}')
            print(f'check[{to_}]={self.dat.check[to_]}', f'begin={from_}')
            # 转移成功
            if len(self.dat.check) > to_ and self.dat.check[to_] == from_:
                self.path.append(i)
                from_ = to_

                self.path.append(from_)
                baseParent = self.dat.base[from_]
                
                # 最后一个字符
                print('compar', f'from={from_}', f'check_val={self.dat.getCheck(baseParent + self.dat.UNUSED_CHAR_VALUE)}')
                print(f'baseParent={baseParent}', f'unused={self.dat.UNUSED_CHAR_VALUE}')
                if self.dat.getCheck(baseParent + self.dat.UNUSED_CHAR_VALUE) == from_:
                    
                    ids = [len(self.path) / 2]
                    k = 0
                    for j in range(1, len(self.path), 2):
                        ids[k] = self.path[j]
                        k += 1
                    
                    # 将 utf8 字符集转化成字符串, 并还原最后字符对应的值
                    self.key = self.dat.charMap.toString(ids)
                    self.value = self.dat.getLeafValue(self.dat.getBase(baseParent + self.bat.UNUSED_CHAR_VALUE))

                    self.path.append(self.dat.UNUSED_CHAR_VALUE)
                    self.currentBase = baseParent

                    return from_

                else:

                    return self.getNext(from_, 0)

            
            return -1
            
        def toString(self):
            return self.key + '=' + self.value


