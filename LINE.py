import random
import numpy as np
from loguru import logger
from tqdm import tqdm

class LINE():
    def __init__(self, dimension: int, order: int = 2):
        self.v_num       = 100000
        self.dim         = int(dimension)
        self.order       = int(order)

        if self.order == 1:
            self.w_vertex_o1 = np.random.rand(self.v_num, self.dim)
        else:
            self.w_vertex = np.random.rand(self.v_num, self.dim)
            self.w_context = np.zeros(self.v_num, self.dim)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, sample_times: int, negative_samples: int, alpha: float):
        logger.info('''
            Model: 
                [LINE]
            
            Learning Parameters:
                sample_times:       {}
                negative_samples:   {}
                alpha:              {}
        '''.format(sample_times, negative_samples, alpha))

        if self.order == 1:
            progress = tqdm(range(sample_times))
            for p in progress:
                # sample v1, v2
                v1 = random.randint(0, self.v_num-1)
                v2 = random.randint(0, self.v_num-1)

                # update
                ## positive
                f = np.dot(self.w_vertex_o1[v1], self.w_vertex_o1[v2])
                f = self.__sigmoid(f)
                g = (1 - f) * alpha
                update_value = np.dot(self.w_vertex_o1[v1], g)
                self.w_vertex_o1[v1] = np.add(self.w_vertex_o1[v1], update_value)
                ## negative
                for _ in range(negative_samples):
                    f = np.dot(self.w_vertex_o1[v1], self.w_vertex_o1[v2])
                    f = self.__sigmoid(f)
                    g = (0 - f) * alpha
                    update_value = np.dot(self.w_vertex_o1[v1], g)
                    self.w_vertex_o1[v1] = np.add(self.w_vertex_o1[v1], update_value)

                # loss
                alpha = alpha * (1 - (p+1)/sample_times)
                progress.set_description("Alpha: {Alpha:.6f}".format(Alpha=alpha))
        else:
            progress = tqdm(range(sample_times))
            for p in progress:
                # sample v1, v2
                v1 = random.randint(0, self.v_num-1)
                v2 = random.randint(0, self.v_num-1)

                # update
                ## positive
                f = np.dot(self.w_vertex[v1], self.w_context[v2])
                f = self.__sigmoid(f)
                g = (1 - f) * alpha
                update_value = np.dot(self.w_vertex[v1], g)
                self.w_vertex[v1] = np.add(self.w_vertex[v1], update_value)
                ## negative
                for _ in range(negative_samples):
                    f = np.dot(self.w_vertex[v1], self.w_context[v2])
                    f = self.__sigmoid(f)
                    g = (0 - f) * alpha
                    update_value = np.dot(self.w_vertex[v1], g)
                    self.w_vertex[v1] = np.add(self.w_vertex[v1], update_value)

                # loss
                alpha = alpha * (1 - (p+1)/sample_times)
                progress.set_description("Alpha: {Alpha:.6f}".format(Alpha=alpha))

line = LINE(0, 1)
line.train(10000, 1, 0.9)