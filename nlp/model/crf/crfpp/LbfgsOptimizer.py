import math

from nlp.model.crf.crfpp.Mcsrch import Mcsrch

"""Limited-memory BFGS 无约束最优化算法，只保存最近的M次迭代信息"""
class LbfgsOptimizer():

    def __init__(self):
        self.w = []
        self.v = []
        self.xi   = []
        self.diag = []

        self.iflag = 0
        self.iscn  = 0
        self.nfev  = 0

        self.iycn  = 0
        self.point = 0
        self.npt   = 0
        self.info  = 0
        self.iter  = 0
        self.ispt  = 0
        self.isyt  = 0
        self.iypt  = 0

        self.maxfev = 0
        self.mcsrch = None
    
    def clear(self):
        self.w = None
        self.v = None
        self.xi   = None
        self.diag = None
        
        self.stp   = 0.0
        self.stp1  = 0.0

        self.iflag = 0
        self.iscn  = 0
        self.nfev  = 0

        self.iycn  = 0
        self.point = 0
        self.npt   = 0
        self.iter  = 0
        self.ispt  = 0
        self.isyt  = 0
        self.iypt  = 0

        self.maxfev = 0
        self.mcsrch = None

    def pseudo_gradient(self, size, v, x, g, C):
        """
        伪梯度函数
        -param size 特征数量大小
        -param v
        -param x 权重
        -param g 特征期望值
        -param C 初始损失值
        """
        for i in range(size):
            if x[i] == 0:
                if g[i] + C < 0:
                    v[i] = g[i] + C
                elif g[i] - C > 0:
                    v[i] = g[i] - C
                else:
                    v[i] = 0

            else:
                v[i] = g[i] + C * Mcsrch.sigma(x[i])

    def lbfgs_optimizer(self, size, msize, x, f, g, diag, w, orthant, C, v, xi, iflag):
        """
        -param size     int     特征数量大小
        -param msize    int     内置参数 默认值 5
        -param x        float[] 最大特征索引组成的数组, 外部的alpha，需要给其设置值，感觉应该是每个特征权重值的存放位置
        -param f        float   损失值
        -param g        float[] 梯度向量
        -param diag     float[] 内置参数, Hessian 矩阵对角线近似
        -param w        float[] 内置参数, 临时向量，存放对角线、步长、梯度差等信息
        -param orthant  boolean 是否使用 L1
        -param C        float   系统设定的初始损失值
        -param v        float[] 内置参数（使用L1时它是具有Size大小的空数组，不使用的时候和 g 的值一样）
        -param xi       float[] 内置参数
        -param iflag    int     内置参数
        """
        yy, ys, bound, cp = 0.0, 0.0, 0, 0
        
        # 默认是False，并不使用L1范数, 当使用L1范数的时候会修改 v 的值
        if orthant:
            self.pseudo_gradient(size, v, x, g, C)
        
        # 定义搜索方法
        if self.mcsrch == None:
            self.mcsrch = Mcsrch()

        firstLoop = True
        
        # initialization
        if iflag == 0:
            self.point = 0
            
            # diag 解释是对角矩阵的近似，表示近似的Hessian矩阵，不过，在这里怎么是向量??？
            # L-BFGS 是limited-memory（有限内存优化）版本，它不像正规得BFGS存储矩阵形式，而是存储部分信息（对角线近似），这种情况下 diag 表示一个向量，每个元素表示Hessian对角上的元素。
            # 在大多数应用中，并不需要整个 Sessian 矩阵，只需要对角部分进行简单操作即可，对角元素可以避免复杂的操作，计算复杂度从O(n^2) 降到 O(n)
            # 初始化一个对称的正定矩阵, 因此对角线上初始化的值全是 1 
            for i in range(size):
                diag[i] = 1.0
            
            # ispt 存储[步长]在 w[] 中的索引; << 左位移 相当于 msize * 2, msize 默认值为 5
            self.ispt = size + (msize << 1)             

            # iypt 存储[梯度差]在 w[] 中的索引；梯度差反映矩阵的方向上的变化
            self.iypt = self.ispt + size * msize       
            
            # 对 w[] 的扩展, 临时存储Hessian矩阵对角线的近似
            # 不过，需要注意的是，这里对角线diag 与 v 进行相乘（只是两个横向量逐一相乘，结果还是向量），具有调整[步长]的作用，反而言之没有处理梯度差
            for i in range(size):
                w[self.ispt + i] = -v[i] * diag[i]
            
            # 计算期望【向量点积】的倒数平方根，即得到一个缩放因子，在调整步长的时候保证数值的稳定
            try:
                self.stp1 = 1.0 / math.sqrt(Mcsrch.ddot(size, v, 0, v, 0))
            except ZeroDivisionError:
                self.stp1 = float('inf')


        # main iteration loop
        while True:
            # 处理每一次迭代(抛开第一次迭代)，更新H矩阵的对角线近似
            # 或是第一次迭代，且标记等于0的时候也可以，这时候进来就是为了处理 当使用L1 范数的时候，重新设置 xi[] 的值
            if not firstLoop or (firstLoop and iflag != 1 and iflag != 2):
                self.iter += 1
                self.info = 0
                
                # 使用 L1 范数
                if orthant:
                    for i in range(size):
                        xi[i] = (x[i] != 0 if Mcsrch.sigma(x[i]) else Mcsrch.sigma(-v[i]))

                if self.iter != 1:
                    if self.iter > size:
                        bound = size
                    
                    # 更新 Hessian 矩阵的对角线近似
                    # computer -h*g using the formula given in: nocedal , j. 1980
                    # ys = 梯度差与步长的点积；yy = 梯度差的点积;
                    ys = Mcsrch.ddot(size, w, self.iypt + self.npt, w, self.ispt + self.npt)
                    yy = Mcsrch.ddot(size, w, self.iypt + self.npt, w, self.iypt + self.npt)

                    for i in range(size):
                        try:
                            diag[i] = ys / yy               # 点积的比值看作是Hessian矩阵对角线项的缩放因子，反映了当前迭代中梯度与步长的相对变化      
                        except ZeroDivisionError:
                            if ys < 0 and yy == 0.0:
                                diag[i] = float("-inf")
                            elif ys == 0.0 and yy == 0.0:
                                diag[i] = float('nan')
                            else:
                                diag[i] = float('inf')


            # 处理非第一次迭代
            if self.iter != 1 and (not firstLoop or (iflag != 1 and not firstLoop)):
                cp = self.point
                if self.point == 0:
                    cp = msize

                w[size + cp - 1] = 1.0 / ys

                
                # 对优化变量的边界进行约束或限制，确保变量保持在一定范围内
                # 在这里具体用于调整步长，决定是否剪裁变量值，或在进行计算搜索方向时对变量进行额外处理？？？
                bound = math.min(self.iter - 1, msize)
                
                # point 默认等于 0 ，后面会累加，最大不会超过mSize
                cp = self.point
                
                # 对[步长]进行裁剪, v 实际上是梯度向量（期望值）
                for i in range(size):
                    w[i] = -v[i]

                for i in range(bound):
                    cp -= 1
                    if cp == -1:
                        cp = msize - 1

                    sq = Mcsrch.ddot(size, w, self.ispt + cp * size, w, 0)
                    inmc = size + msize + cp
                    iycn = self.iypt + cp * size

                    w[inmc] = w[size + cp] * sq
                    d = -w[inmc]
                    
                    # 对 w 进行线性计算
                    Mcsrch.daxpy(size, d, w, iycn, w, 0)
                
                # 对[方向]进行裁剪
                for i in range(size):
                    w[i] = diag[i] * w[i]

                for i in range(bound):
                    yr = Mcsrch.ddot(size, w, self.iypt + cp * size, w, 0)
                    beta = w[size + cp] * yr

                    inmc = size + msize + cp
                    beta = w[inmc] - beta
                    iscn = self.ispt + cp * size

                    Mcsrch.daxpy(size, beta, w, iscn, w, 0)

                    cp += 1
                    if cp == msize:
                        cp = 0

                if orthant:
                    for i in range(size):
                        w[i] = Mcsrch.sigma(w[i]) == Mcsrch.sigma(-v[i]) if w[i] else 0
                
                # store the new search direction
                for i in range(size):
                    w[self.ispt + self.point * size + i] = w[i]
            
            # obtain the one-dimensional minimizer of the function
            # by using the line search routine Mcsrch
            if not firstLoop or (firstLoop and iflag != 1):
                self.nfev = 0
                self.stp  = 1.0

                if self.iter == 1:
                    self.stp = self.stp1
                
                # 重新对 w 前面的数据(范围: size )进行赋值，注意下面的梯度变化率的计算与它有关
                # 在w[] 初始化的时候都为 0，刚开始也只是从lspt开始赋值
                for i in range(size):
                    w[i] = g[i]
            
            # 线性搜索，确定步长
            stpArr  = [self.stp]    # 步长
            infoArr = [self.info]   # 诊断信息，表示算法状态和退出信息； 状态有：[-1, 0, 1, 2, 3, 4, 5, 6] 这些是mcsrch()算法定义的状态
            nfevArr = [self.nfev]   # 表示目标函数被调用次数，衡量算法计算成本和收敛速度
            
            self.mcsrch.mcsrch(size, x, f, v, w, self.ispt + self.point * size, stpArr, infoArr, nfevArr, diag)

            self.stp  = stpArr[0]
            self.info = infoArr[0]
            self.nfev = nfevArr[0]
            
            # 在这里给 alpha[] 赋值, 但是，也仅限与当使用L1范数的时候进行赋值，alpha[] 的值是 0|1 构成的
            if self.info == -1:
                if orthant:
                    for i in range(size):
                        x[i] = Mcsrch.sigma(x[i]) == Mcsrch.sigma(xi[i]) if x[i] else 0

                return 1


            if self.info != -1:
                print('The line search routine mcsrch failed: error code:' + self.info)
                return -1

            # compute the new step and gradient change
            # 计算新的梯度信息到 w[] 
            self.npt = self.point * size
            for i in range(size):
                w[self.ispt + self.npt + i] = self.stp * w[self.ispt + self.npt + i]
                w[self.iypt + self.npt + i] = g[i] - w[i]

            self.point += 1
            if self.point == msize:
                self.point = 0
            
            # v 表示期望值，整个算法中不变；x 表示模型参数，会在mcsrch中更新
            gnorm = math.sqrt(Mcsrch.ddot(size, v, 0, v, 0))
            xnorm = max(1.0, math.sqrt(Mcsrch.ddot(size, x, 0, x, 0)))
            
            # 如果说模型参数对比期望值比例很小，说明模型参数变化不再显著，模型趋于收敛
            if (gnorm / xnorm) <= Mcsrch.eps:
                return 0

            firstLoop = False


    def optimize(self, size, x, f, g, orthant = False, C = 1.0):
        """
        优化器
        -param size     int         特征数量大小
        -param x        float[]     最大特征索引组成的数组 
        -param f        float       损失值
        -param g        float[]     梯度向量
        -param orthant  boolean     是否使用 L1 范数，默认 L2
        -param C        float       系统设定的损失值
        """
        
        # 进行一些初始化工作
        msize = 5

        if not self.w:
            self.iflag = 0
            self.w = [0.0] * (size * (2 * msize + 1) + 2 * msize)
            self.diag = [0.0] * size
            self.v = [0.0] * size

            if orthant:
                self.xi = [0.0] * size

        elif len(self.diag) != size or len(self.v) != size:
            return -1

        elif orthant and len(self.v) != size:
            return -1

        _iflag = 0
        if orthant:
            _iflag = self.lbfgs_optimizer(size, msize, x, f, g, self.diag, self.w, orthant, C, self.v, self.xi, self.iflag)
        else:
            _iflag = self.lbfgs_optimizer(size, msize, x, f, g, self.diag, self.w, orthant, C, g, self.xi, self.iflag)
        
        self.iflag = _iflag

        if _iflag < 0:
            return -1

        if _iflag == 0:
            self.clear()
            return 0

        return 1


