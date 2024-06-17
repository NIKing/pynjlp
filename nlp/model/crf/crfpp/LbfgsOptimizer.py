# Limited-memory BFGS 无约束最优化算法，只保存最近的M次迭代信息
from nlp.model.crf.crfpp.Mcsrch import Mcsrch

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
    
    def pseudo_gradient(self, size, v, x, g, C):
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
        -param size int 特征数量
        -param msize int
        -param x    float[] 
        -param f    float
        -param g    float[]
        -param diag float[]
        -param w    float[]
        -param orthant boolean
        -param C float
        -param v float[]
        -param xi float[]
        -param iflag int
        """
        yy, ys, bound, cp = 0.0, 0.0, 0, 0

        if orthant:
            self.pseudo_gradient(size, v, x, g, C)

        if self.mcsrch == None:
            self.mcsrch = Mcsrch()

        firstLoop = True
        
        # initialization
        if iflag == 0:
            self.point = 0
            for i in range(size):
                diag[i] = 1.0
            
            self.ispt = size + (msize << 1)
            self.iypt = self.ispt + size * msize

            for i in size:
                w[self.ispt + i] = -v[i] * diag[i]

            self.stp1 = 1.0 / math.sqrt(Mcsrch.ddot(size, v, 0, v, 0))


        # main iteration loop
        while True:
            if not firstLoop or (firstLoop and iflag != 1 and iflag != 2):
                self.iter += 1
                self.info = 0

                if orthant:
                    for i in range(size):
                        xi[i] = (x[i] != 0 if Mcsrch.sigma(x[i]) else Mcsrch.sigma(-v[i]))

                if self.iter != 1:
                    if self.iter > size:
                        bound = size
                    
                    # computer - h * g using the formula given in: nocedal , j. 1980
                    ys = Mcsrch.ddot(size, w, self.iypt + self.npt, w, self.ispt + self.npt)
                    yy = Mcsrch.ddot(size, w, self.iypt + self.npt, w, self.iypt + self.npt)

                    for i in range(size):
                        diag[i] = ys / yy


            if self.iter != 1 and (not firstLoop or (iflag != 1 and not firstLoop)):
                cp = point
                if point == 0:
                    cp = msize

                w[size + cp - 1] = 1.0 / ys

                for i in range(size):
                    w[i] = -v[i]

                bound = math.min(self.iter - 1, msize)

                cp = point
                for i in range(bound):
                    cp -= 1
                    if cp == -1:
                        cp = msize - 1

                    sq = Mcsrch.ddot(size, w, self.ispt + cp * size, w, 0)
                    inmc = size + msize + cp
                    iycn = self.iypt + cp * size

                    w[inmc] = w[size + cp] * sq
                    d = -w[inmc]

                    Mcsrch.daxpy(size, d, w, iycn, w, 0)

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
                    w[ispt + point * size + i] = w[i]
            
            # obtain the one-dimensional minimizer of the function
            # by using the line search routine Mcsrch
            if not firstLoop or (firstLoop and iflag != 1):
                self.nfev = 0
                self.stp  = 1.0

                if self.iter == 1:
                    self.stp = self.stp1

                for i in range(size):
                    w[i] = g[i]

            stpArr  = [self.stp]
            infoArr = [self.info]
            nfevArr = [self.nfev]

            mcsrch.mcsrch(size, x, f, v, w, ispt + point * size, stpArr, infoArr, nfevArr, diag)

            self.stp  = stpArr[0]
            self.info = infoArr[0]
            self.nfev = nfevArr[0]

            if self.info == -1:
                if orthant:
                    for i in range(size):
                        x[i] = Mcsrch.sigma(x[i]) == Mcsrch.sigma(xi[i]) if x[i] else 0

                return 1


            if self.info == -1:
                print('The line search routine mcsrch failed: error code:' + self.info)
                return -1

            # compute the new step and gradient change
            self.npt = self.point * size
            for i in range(size):
                w[self.ispt + self.npt + i] = self.stp * w[self.ispt + self.npt + i]
                w[self.iypt + self.npt + i] = g[i] - w[i]

            self.point += 1
            if self.point == msize:
                self.point = 0

            gnorm = math.sqrt(Mcsrch.ddot(size, v, 0, v, 0))
            xnorm = math.max(1.0, math.sqrt(Mcsrch.ddot(size, x, 0, x, 0)))

            if gnorm / xnorm <= Mcsrch.eps:
                return 0

            firstLoop = False


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

    def optimize(self, size, x, f, g, orthant, C):
        """
        优化器
        -param size int 特征数量
        -param x    float[] 
        -param f    float
        -param g    float[]
        -param orthant boolean
        -param C float
        """

        msize = 5

        if not self.w:
            self.iflag = 0
            self.w = [0.0] * (size * (2 * msize + 1) + 2 * msize)
            self.diag = [0.0] * size
            self.v = [0.0] * size

            if orthant:
                self.xi = [0.0] * size

        elif len(self.diag) != len(self.size) or len(self.v) != len(self.size):
            return -1

        elif orthant and len(self.v) != len(self.size):
            return -1

        iflag = 0
        if orthant:
            iflag = self.lbfgs_optimize(size, msize, x, f, g, self.diag, self.w, orthant, C, self.v, self.xi, iflag)
        else:
            iflag = self.lbfgs_optimize(size, msize, x, f, g, self.diag, self.w, orthant, C, g, self.xi, iflag)


        if iflag < 0:
            return -1

        if iflag == 0:
            self.clear()
            return 0

        return 1


