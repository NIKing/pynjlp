from .HashCode import hash_code

class Node:
    def __init__(self):
        self.code = 0
        self.depth = 0
        self.left = 0
        self.right = 0

    def toString(self):
        return "Node{" \
                "code=" + self.code \
                ",depth=" + self.depth \
                ",left="  + self.left \
                ",right=" + self.right
                "}"

class DoubleArrayTrie:

    def __init__(self, buildFrom, enableFastBuild = False):
        self.check = []
        self.base  = []

        self.key = []
        self.value = []
        self.length = []

        self.size = 0
        self.keySize = 0
        self.allocSize = 0
        self.error = 0

        self.progress = 0
        self.nextCheckPos = 0  # 下一次检查的位置

        self.enableFastBuild = enableFastBuild 

        if self.build(buildFrom) != 0:
            print("构造失败")
    
    def resize(self, newSize):
        """拓展数组"""
        base2  = [0] * newSize
        check2 = [0] * newSize
        
        if self.allocSize > 0:
            base2   = self.base[:self.allocSize]
            check2  = self.check[:self.allocSize]
        
        self.base  = base2
        self.check = check2
        self.allocSize = newSize

    def build(self, keyValueMap):
        assert keyValueMap != null
        items = keyValueMap.items()
        
        return build_items(items)

    def build_items(self, keyList, valueList):
        assert len(keyList) == len(valueList) : "键的个数与值的个数不一样！"
        assert len(keyList) > 0 : "键值个数为0！"

        return build_keys(keyList, None, None, len(keyList))

    def build_keys(self, _key, _length, _value, _keySize):
        """
        -param list _key 必须是字典序
        -param list _length 对应每个key的长度，留空动态获取
        -param list _value 每个key对应的值，留空使用key的下标作为值
        -param int key的长度，应该设为_key, size
        return 是否出错
        """

        if not _key or _keySize > len(_key):
            return 0

        self.key = _key
        self.length = _length
        self.keySize = _keySize
       
        # 32个双字节
        self.resize(65536 * 32)

        self.base[0] = 1
        self.nextCheckPos = 0
        self.progress = 0
        
        # 根节点
        root_node = Node()
        root_node.left = 0
        root_node.right = _keySize
        root_node.depth = 0

        siblings = self.fetch(root_node)
        self.insert(siblings, {})
        self.shrink()

        return self.error

    def fetch(self, parent) -> list:
        """
        获取直接相连的子节点, 没有就创建节点
        -param parent 父节点
        -return 兄弟节点列表
        """
        if self.error < 0:
            return []
        
        siblings = []
        prev, i = 0, parent.left
        while i < parent.right:

            length = self.length[i] if self.length != None else len(self.key[i])
            if length < parent.depth:
                continue

            tmp = self.key[i]

            cur = 0
            length = self.length[i] if self.length != None else len(tmp)
            if length != parent.depth:
                # 在java中使用 (int) 获取字符的Unicode值，在这里使用hash_code()获取
                cur = hash_code(tmp[parent.depth]) + 1

            if prev > cur:
                self.error = -3
                return []

            if cur != prev:
                tmp_node = Node()
                tmp_node.depth = parent.depth + 1
                tmp_node.code  = cur
                tmp_node.left  = i
                
                if len(siblings) != 0:
                    siblings[len(siblings) - 1].right = i

                siblings.add(tmp_node)

            prev = cur
            i += 1

        if len(siblings) != 0:
            siblings[len(siblings) - 1].right = parent.right

        return siblings
                

    
    def insert(self, siblings, used):
        """
        插入节点
        -param list siblings 等待插入的兄弟节点
        -return int 插入位置
        """
        if self.error < 0:
            return 0

        begin = 0
        pos = max(siblings[0].code + 1, (self.nextCheckPos + 1) if self.enableFastBuild else self.nextCheckPos) - 1
        nonzero_num = 0
        first = 0

        if self.allocSize <= pos:
            self.resize(pos + 1)
    
        """
        扩容策略
        此循环的目标是找出满足check[begin + a1...an] == 0 的 n 个空闲空间, a1...an是siblings的n个节点
        """
        while True:
            pos++

            if self.allocSize <= pos:
                self.resize(pos + 1)

            if self.check[pos] != 0:
                nonzero_num++
                continue

            if first == 0:
                nextCheckPos = pos
                first = 1

            begin = pos - siblings[0].code # 当前位置离第一个兄弟节点的距离
            if self.allocSize <= (begin + siblings[len(siblings) - 1].code):
                self.resize(begin + siblings[len(siblings) - 1].code + 65535)

            if used.get(begin, 0):
                continue
            
            # 这种循环方式是为了实现 java 中的 continue outer；当内循环正常终止，才终止外循环，否则继续外循环
            for i in range(len(siblings)):
                if self.check[begin + siblings[i].code] != 0:
                    break
            else:
                continue

            break

        
        """
        启发式空闲搜索法
        从位置nextCheckPos开始到pos之间，如果占用率在95%以上，下次插入节点时，直接从pos位置开始查找 
        """ 
        if 1.0 * nozero_num / (pos - self.nextCheckPos + 1) >= 0.95:
            self.nextCheckPos = pos



        used.setDefault(begin, 1)

        if self.size <= begin + siblings[len(siblings) - 1].code + 1:
            self.size = begin + siblings[len(siblings) - 1].code + 1

        # 检查每个子节点 
        # 给 check 赋值，建立子节点 (check[begin + s.code]) 与父节点 (begin) 的多对一的关系
        # check的对应位置是：字符hash值 + 1 ，其赋的值是：上一个字符的位置(父节点)，初始等于1
        for i in range(len(siblings)):
            self.check[begin + siblings[i].code] = begin
        
        # 检查每个子节点, 若其没有孩子，就将它的base设置 -1，否则就调用 insert 建立关系 
        # 给 base 赋值，建立父节点 (base[begin + s.code]) 与子节点 (h) 的一对多的关系
        for i in range(len(siblings)):
            new_siblings = self.fetch(siblings[i])
            
            # 一个词的终止且不为其他词的前缀
            if len(new_siblings) == 0:
               self.base[begin + siblings[i].code] = (-self.value[siblings[i].left] - 1) if self.value else (-siblings[i].left - 1)

               if self.value and (-self.value[i].left - 1) >= 0:
                   self.error = 2
                   return 0

               self.progress++

            else:
                h = self.insert(new_siblings, used)
                self.base[begin + siblings[i].code] = h

        return begin




