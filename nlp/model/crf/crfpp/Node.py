class Node():

    LOG2 = 0.69314718055
    MINUS_LOG_EPSILON = 50

    def __init__(self):
        self.x = 0
        self.y = 0

        self.alpha = 0.0
        self.beta  = 0.0
        self.cost  = 0.0
        self.bestCost = 0.0

        self.prev = None

        self.fVector = []
        self.lpath = []
        self.rpath = []




