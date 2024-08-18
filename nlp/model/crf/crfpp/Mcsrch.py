import math

"""模糊搜索"""
class Mcsrch():
    ftol = 1e-4
    xtol = 1e-16
    eps  = 1e-7
    
    lb3_1_gtol      = 0.9
    lb3_1_stpmin    = 1e-20
    lb3_1_stpmax    = 1e20

    lb3_1_mp    = 6
    lb3_1_lp    = 6

    def __init__(self):
        self.infoc = 0
        self.stage1 = False
        self.brackt = False

        self.finit  = 0.0
        self.dginit = 0.0
        self.dgtest = 0.0

        self.width  = 0.0
        self.width1 = 0.0

        self.stx    = 0.0
        self.fx     = 0.0
        self.dgx    = 0.0
        
        self.sty    = 0.0
        self.fy     = 0.0
        self.dgy    = 0.0
        
        self.stmin  = 0.0
        self.stmax  = 0.0


    def mcsrch(self, size, x, f, g, s, startOffset, stp, info, nfev, wa):
        p5 = 0.5
        p66 = 0.66
        xtrapf = 4.0
        maxfev = 20

        if info[0] != -1:
            self.infoc = 1

            if size <= 0 or stp[0] <= 0.0:
                return

            dginit = Mcsrch.ddot(size, g, 0, s, startOffset)
            if dginit >= 0.0:
                return

            nfev[0]= 0
            
            self.brackt = False
            self.stage1 = True
            self.finit  = f

            self.dgtest = Mcsrch.ftol * dginit
            self.width  = Mcsrch.lb3_1_stpmax - Mcsrch.lb3_1_stpmin
            self.width1 = self.width / p5

            for j in range(size):
                wa[j] = x[j]

            self.stx = 0.0
            self.fx  = self.finit
            self.dgx = self.dginit
            
            self.sty = 0.0
            self.fy  = self.finit
            self.dgy = self.dginit
        
        firstLoop = True
        while True:

            if not firstLoop or (firstLoop and info[0] != -1):

                if self.brackt:
                    self.stmin = min(self.stx, self.sty)
                    self.stmax = max(self.stx, self.sty)
                else:
                    self.stmin = self.stx
                    self.stmax = stp[0] + xtrapf * (stp[0] - self.stx)
                
                stp[0] = max(stp[0], Mcsrch.lb3_1_stpmin)
                stp[0] = min(stp[0], Mcsrch.lb3_1_stpmax)

                if (
                        self.brackt and ( \
                            (stp[0] <= self.stmin or stp[0] >= self.stmax) or \
                            nfev[0] >= maxfev - 1 or \
                            self.infoc == 0 ) \
                    ) or ( \
                        self.brackt and (self.stmax - self.stmin <= Mcsrch.xtol * self.stmax) \
                    ):
                        stp[0] = self.stx
                
                for j in range(size):
                    x[j] = wa[j] + stp[0] * s[startOffset + j]

                info[0] = -1

                return

            info[0] = 0
            nfev[0] += 1

            dg = Mcsrch.ddot(size, g, 0, s, startOffset)
            ftest1 = self.finit + stp[0] * self.dgtest

            if self.brackt and ((stp[0] <= self.stmin or stp[0] >= self.stmax) or self.infoc == 0):
                info[0] = 6
            
            if stp[0] == Mcsrch.lb3_1_stpmax and f <= ftest1 and dg <= self.dgtest:
                info[0] = 5

            if stp[0] == Mcsrch.lb3_1_stpmin and (f > ftest1 or dg >= self.dgtest):
                info[0] = 4
            
            if nfev[0] >= maxfev:
                info[0] = 3
            
            if self.brackt and self.stmax - self.stmin <= Mcsrch.xtol * self.stmax:
                info[0] = 2
            
            if f <= ftest1 and abs(dg) <= Mcsrch.lb3_1_gtol * (-self.dginit):
                info[0] = 1
            
            if info[0] != 0:
                return
            
            if self.stage1 and f <= ftest1 and dg >= min(Mcsrch.ftol, Mcsrch.lb3_1_gtol) * self.dginit:
                self.stage1 = False
            
            if self.stage1 and f <= self.fx and f > ftest1:
                fm = f - stp[0] * self.dgtest

                fxm = self.fx - self.stx * self.dgtest
                fym = self.fy - self.sty * self.dgtest
                dgm = dg - self.dgtest
                dgxm = self.dgx - self.dgtest
                dgym = self.dgy - self.dgtest

                stxArr = [self.stx]
                fxmArr = [fxm]
                dgxmArr = [dgxm]

                styArr = [self.sty]
                fymArr = [fym]
                dgymArr = [dgym]
                
                bracktArr = [self.brackt]
                infocArr = [self.infoc]
                
                Mcsrch.mcstep(stxArr, fxmArr, dgxmArr, styArr, fymArr, dgymArr, stp, fm, dgm, bracktArr,
                       self.stmin, self.stmax, infocArr)

                self.stx = stxArr[0]
                fxm = fxmArr[0]
                dgxm = dgxmArr[0]
                
                self.sty = styArr[0]
                fym = fymArr[0]
                dgym = dgymArr[0]
                
                self.brackt = bracktArr[0]
                self.infoc = infocArr[0]

                self.fx = fxm + self.stx * self.dgtest
                self.fy = fym + self.sty * self.dgtest
                self.dgx = dgxm + self.dgtest
                self.dgy = dgym + self.dgtest

            else:

                stxArr = [self.stx]
                fxArr = [self.fx]
                dgxArr = [self.dgx]

                styArr = [self.sty]
                fyArr = [self.fy]
                dgyArr = [self.dgy]

                bracktArr = [self.brackt]
                infocArr = [self.infoc]
                
                Mcsrch.mcstep(stxArr, fxArr, dgxArr, styArr, fyArr, dgyArr, stp, f, dg, bracktArr,
                       self.stmin, self.stmax, infocArr)
                
                self.stx = stxArr[0]
                self.fx = fxArr[0]
                self.dgx = dgxArr[0]

                self.sty = styArr[0]
                self.fy = fyArr[0]
                self.dgy = dgyArr[0]
                
                self.brackt = bracktArr[0]
                self.infoc = infocArr[0]
            
            if self.brackt:
                d1 = self.sty - self.stx
                if abs(d1) >= p66 * self.width1:
                    stp[0] = self.stx + p5 * (self.sty - self.stx)

                self.width1 = self.width

                d1 = self.sty - self.stx
                self.width = abs(d1)
            
            firstLoop = False


    @staticmethod
    def sigma(x) -> float:
        """
        符号函数，返回输入值的符号
        -param x float 
        """
        if x > 0:
            return 1.0
        elif x < 0:
            return -1.0

        return 0.0
    
    @staticmethod
    def ddot(size, dx, offsetX, dy, offsetY) -> float:
        """
        向量的点积计算
        -param size     向量的长度
        -param dx       向量x 
        -param offsetX  向量x的偏移量
        -param dy       向量y
        -param offsetY  向量y的偏移量
        """
        res = 0.0

        for i in range(size):
            res += dx[i + offsetX] * dy[i + offsetY]

        return res

    @staticmethod
    def mcstep(stx, fx, dx, 
            sty, fy, dy, 
            stp, fp, dp, 
            brackt, 
            stpmin, stpmax, 
            info):
        bound = True
        info[0] = 0
        
        p, q, s, d1, d2, d3, r, gamma, theta, stpq, stpc, stpf = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if brackt[0] and ( \
                (stp[0] <= min(stx[0], sty[0]) or stp[0] >= max(stx[0], sty[0])) or \
                dx[0] * (stp[0] - stx[0]) >= 0.0 or stpmax < stpmin \
            ):
            return
        
        sgnd = dp * (dx[0] / abs(dx[0])) if dx[0] != 0 else 0
        
        if fp > fx[0]:
            info[0] = 1
            bound = True
            theta = (fx[0] - fp) * 3 / (stp[0] - stx[0]) + dx[0] + dp
            
            d1 = abs(theta)
            d2 = abs(dx[0])
            
            d1 = max(d1, d2)
            d2 = abs(dp)
            s  = max(d1, d2)
            
            d1 = theta / s

            gamma = s * math.sqrt(d1 * d1 - dx[0] / s * (dp / s))

            if stp[0] < stx[0]:
                gamma = -gamma
            
            p = gamma - dx[0] + theta
            q = gamma - dx[0] + gamma + dp
            
            r = p / q
            stpc = stx[0] + r * (stp[0] - stx[0])
            stpq = stx[0] + dx[0] / ((fx[0] - fp) /
                    (stp[0] - stx[0]) + dx[0]) / 2 * (stp[0] - stx[0])

            d1 = stpc - stx[0]
            d2 = stpq - stx[0]
            if abs(d1) < abs(d2):
                stpf = stpc
            else:
                stpf = stpc + (stpq - stpc) / 2

            brackt[0] = True

        elif sgnd < 0.0:
            info[0] = 2
            bound = False
            theta = (fx[0] - fp) * 3 / (stp[0] - stx[0]) + dx[0] + dp
            
            d1 = abs(theta)
            d2 = abs(dx[0])
            d1 = max(d1, d2)
            d2 = abs(dp)
            s = max(d1, d2)
            d1 = theta / s
            gamma = s * math.sqrt(d1 * d1 - dx[0] / s * (dp / s))

            if stp[0] > stx[0]:
                gamma = -gamma

            p = gamma - dp + theta
            q = gamma - dp + gamma + dx[0]
            r = p / q
            
            stpc = stp[0] + r * (stx[0] - stp[0])
            stpq = stp[0] + dp / (dp - dx[0]) * (stx[0] - stp[0])
            
            d1 = stpc - stp[0]
            d2 = stpq - stp[0]
            
            if abs(d1) > abs(d2):
                stpf = stpc
            else:
                stpf = stpq
            
            brackt[0] = True
        
        elif abs(dp) < abs(dx[0]):
            info[0] = 3
            bound = True
            theta = (fx[0] - fp) * 3 / (stp[0] - stx[0]) + dx[0] + dp
            
            d1 = abs(theta)
            d2 = abs(dx[0])
            d1 = max(d1, d2)
            d2 = abs(dp)
            s = max(d1, d2)
            d3 = theta / s
            d1 = 0.0
            d2 = d3 * d3 - dx[0] / s * (dp / s)
            gamma = s * math.sqrt((max(d1, d2)))

            if stp[0] > stx[0]:
                gamma = -gamma

            p = gamma - dp + theta
            q = gamma + (dx[0] - dp) + gamma
            r = p / q

            if r < 0.0 and gamma != 0.0:
                stpc = stp[0] + r * (stx[0] - stp[0])
            elif stp[0] > stx[0]:
                stpc = stpmax
            else:
                stpc = stpmin

            stpq = stp[0] + dp / (dp - dx[0]) * (stx[0] - stp[0])
            if brackt[0]:
                d1 = stp[0] - stpc
                d2 = stp[0] - stpq
                
                if abs(d1) < abs(d2):
                    stpf = stpc
                else:
                    stpf = stpq
            else:
                d1 = stp[0] - stpc
                d2 = stp[0] - stpq
                
                if abs(d1) > abs(d2):
                    stpf = stpc
                else:
                    stpf = stpq

        else:
            info[0] = 4
            bound = False
            
            if brackt[0]:
                theta = (fp - fy[0]) * 3 / (sty[0] - stp[0]) + dy[0] + dp
                d1 = abs(theta)
                d2 = abs(dy[0])
                d1 = max(d1, d2)
                d2 = abs(dp)
                s = max(d1, d2)
                d1 = theta / s
                gamma = s * math.sqrt(d1 * d1 - dy[0] / s * (dp / s))
                
                if stp[0] > sty[0]:
                    gamma = -gamma
                
                p = gamma - dp + theta
                q = gamma - dp + gamma + dy[0]
                r = p / q
                
                stpc = stp[0] + r * (sty[0] - stp[0])
                stpf = stpc
            
            elif stp[0] > stx[0]:
                stpf = stpmax
            
            else:
                stpf = stpmin

        if fp > fx[0]:
            sty[0] = stp[0]
            fy[0] = fp
            dy[0] = dp
        else:
            if sgnd < 0.0:
                sty[0] = stx[0]
                fy[0] = fx[0]
                dy[0] = dx[0]
            
            stx[0] = stp[0]
            fx[0] = fp
            dx[0] = dp

        stpf = min(stpmax, stpf)
        stpf = max(stpmin, stpf)
        stp[0] = stpf

        if brackt[0] and bound:
        
            if (sty[0] > stx[0]):
                d1 = stx[0] + (sty[0] - stx[0]) * 0.66
                stp[0] = min(d1, stp[0])
            else:
                d1 = stx[0] + (sty[0] - stx[0]) * 0.66
                stp[0] = max(d1, stp[0])

        return


