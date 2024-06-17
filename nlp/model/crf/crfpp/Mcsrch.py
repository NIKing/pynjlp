class Mcsrch():
    
    ftol = 1e-4,
    xtol = 1e-16,
    eps  = 1e-7,
    
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
        self.dgx    = 0.0
        self.stmin  = 0.0
        self.stmax  = 0.0


    def sigma(x) -> float:
        """
        正弦函数
        -param x float 
        """
        if x > 0:
            return 1.0
        elif x < 0:
            return -1.0

        return 0.0
