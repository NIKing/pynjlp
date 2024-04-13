from queue import PriorityQueue

class MaxHeap:
    def __init__(self, maxSize, comparator):
        if maxSize <= 0:
            raise ValueError()

        self.maxSize = maxSize
        self.comparator = comparator

        self.queue = PriorityQueue(maxsize = maxSize)


    def add(self, e) -> bool:
        """添加一个元素 e """

        # 未达到最大容量，直接添加
        if self.queue.size() < self.maxSize:
            self.queue.put(e)
            return True
        
        # 队列已满，将新元素与当前堆顶元素比较，保留较小的元素
        else:
            peek = self.queue.peek()
            
            if self.comparator(e, peek) > 0:
                self.queue.poll()
                self.queue.put(e)

                return True

        return False

    def addAll(self, collection):
        for e in collection:
            self.add(e)

        return self

    def toList(self):
        _list = []

        while not self.queue.isEmpty():
            _list.append(self.queue.poll())

        return _list


    def iterator(self):
        return self.queue.iterator()

    def size(self):
        return self.queue.size()

