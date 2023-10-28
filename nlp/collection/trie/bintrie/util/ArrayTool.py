
def binarySearch(branches, c):
    """
    二分树查找-（有序查找）
    二分树查找的对象必须是有序队列，也就是把需要查找的队列不断的一分为二查找。
    算法需要两个关键信息：
     1，一个是数列的中间位置 mid_pos
     2，一个是数列的中间位置的值 mid_val 
    
    在算法执行之前，需要定义两个变量：
     1，左指针，初始指数列的最左边数字，在这里指 low，初始等于0
     2，右指针，初始指数列的最右边数字，在这里指队列长度 high
    """

    high = len(branches) - 1
    if len(branches) < 1:
        return high

    low = 0
    while low <= high:
        #mid = (low + high) >>> 1
        # // 整数除法运算，结果只会是整数部分
        mid = (low + high) // 2
        cmp = branches[mid].compareTo(c)
        
        # 说明，应该把查找范围缩小在左边，反之在右边
        if cmp < 0:
            low = mid + 1
        elif cmp > 0:
            high = mid - 1
        else:
            return mid

    return -(low + 1)

