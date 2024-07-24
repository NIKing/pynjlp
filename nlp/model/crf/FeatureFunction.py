class FeatureFunction():

    def __init__(self, o = '', tagSize = 0):
        self.o = o                  # 标签
        self.w = [0.0] * tagSize    # 权值，按照index对应于tag的id
