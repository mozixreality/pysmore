from random import random

MONITOR            = 10000
SIGMOID_TABLE_SIZE = 1000
MAX_SIGMOID        = 8.0

class Vertex():
    def __init__(self):
        self.offset = 0
        self.branch = 0
        self.out_degree = 0.0
        self.in_degree  = 0.0

class AliasTable():
    def __init__(self):
        self.alias = -1
        self.prob  = 0.0

class ProNet():
    def __init__(self):
        # MAX index number
        self.MAX_line = 0
        self.MAX_vid  = 0

        # Alias Graph
        self.vertex    = []
        self.vertex_AT = []

        # Cahce
        self.cached_sigmoid = []

    def fast_sigmoid(self, x: float):
        if x < -MAX_SIGMOID:
            return 0.0
        elif x > MAX_SIGMOID: 
            return 1.0
        else:
            return self.cached_sigmoid[ int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2) ]

    def source_sample(self):
        rand_p = random.randint(0, 1)
        rand_v = random.randint(0, self.MAX_vid)
        if rand_p < self.vertex_AT[rand_v].prob:
            return rand_v
        else:
            return self.vertex_AT[rand_v].alias

    def target_sample(self, vid: int):
        if self.vertex[vid].branch == 0:
            return -1

        rand_p = random.randint(0, 1)
        rand_v = random.randint(0, self.vertex[vid].branch) + self.vertex[vid].offset

        if rand_p < self.context_AT[rand_v].prob:
            return self.context[rand_v].vid
        else:
            return self.context_AT[rand_v].alias

    def update_pair(self, w_vertex, w_context, vertex: int, context: int, dimension: int, negative_samples: int, alpha: float):
        back_err = [0.0] * dimension
        pass

    def opt_sigmoidSGD(self, w_vertex_v, w_context_c, label: float, dimension: int, alpha: float, back_err, loss_context_c):
        pass