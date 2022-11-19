import random
import numpy as np
from loguru import logger
from tqdm import tqdm

SEED = 0

class LINE():
    def __init__(self, dimension: int, order: int = 2):
        np.random.seed(SEED)

        v_num, positive_samples, itm_to_vid, vid_to_itm, usr_to_vid, vid_to_usr = self.__load_data__()
        self.v_num      = v_num
        self.dim        = int(dimension)
        self.order      = int(order)

        self.pos_lst    = positive_samples # usr to itm
        self.itm_to_vid = itm_to_vid
        self.vid_to_itm = vid_to_itm
        self.usr_to_vid = usr_to_vid
        self.vid_to_usr = vid_to_usr

        if self.order == 1:
            self.w_vertex_o1 = np.random.rand(self.v_num, self.dim)
        else:
            self.w_vertex = np.random.rand(self.v_num, self.dim)
            self.w_context = np.zeros(self.v_num, self.dim)

        logger.info("Model: [LINE], Total Vertex: {}".format(self.v_num))

    def __sample_neg__(self):
        return random.randint(0, self.v_num-1)

    def __sample_src__(self):
        return random.choice(list(self.pos_lst))

    def __sample_tar__(self, v1):
        return random.choice(self.pos_lst[v1])

    def __load_data__(self):
        logger.info('loading data')

        v_num = 0
        itm_to_vid = {}
        vid_to_itm = {}
        usr_to_vid = {}
        vid_to_usr = {}
        positive_samples = {}

        with open("./ml-1m/movies.csv") as file:
            line = file.readline()
            line = file.readline()

            while(line):
                mov = line.split(",")[0]
                mov = int(mov)

                if mov not in itm_to_vid:
                    itm_to_vid[mov] = v_num
                    vid_to_itm[v_num] = mov
                    v_num += 1

                line = file.readline()

        with open("./ml-1m/ratings.csv") as file:
            line = file.readline()
            line = file.readline()

            while(line):
                usr, mov, rate, _ = line.split(",")
                usr = int(usr)
                mov = int(mov)

                if usr not in usr_to_vid:
                    usr_to_vid[usr] = v_num
                    vid_to_usr[v_num] = usr
                    v_num += 1
                   
                if float(rate) >= 3.0:
                    if usr_to_vid[usr] not in positive_samples:
                        positive_samples[usr_to_vid[usr]] = [itm_to_vid[mov]]
                    else:
                        positive_samples[usr_to_vid[usr]].append(itm_to_vid[mov])

                line = file.readline()

        return v_num, positive_samples, itm_to_vid, vid_to_itm, usr_to_vid, vid_to_usr

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
                v1 = self.__sample_src__()
                v2 = self.__sample_tar__(v1)

                # update
                ## positive
                f = np.dot(self.w_vertex_o1[v1], self.w_vertex_o1[v2])
                f = 1 / (1 + np.exp(-f))
                g = (1 - f) * alpha
                update_value = np.dot(self.w_vertex_o1[v1], g)
                total_update_value = update_value
                self.w_vertex_o1[v2] = np.add(self.w_vertex_o1[v2], update_value)
                ## negative
                for _ in range(negative_samples):
                    v2 = self.__sample_neg__()
                    f = np.dot(self.w_vertex_o1[v1], self.w_vertex_o1[v2])
                    f = 1 / (1 + np.exp(-f))
                    g = (0 - f) * alpha
                    update_value = np.dot(self.w_vertex_o1[v1], g)
                    total_update_value =  np.add(total_update_value, update_value)
                    self.w_vertex_o1[v2] = np.add(self.w_vertex_o1[v2], update_value)
                self.w_vertex_o1[v1] = np.add(self.w_vertex_o1[v1], total_update_value)

                # loss
                alpha = alpha * (1 - (p+1)/sample_times)
                progress.set_description("Alpha: {Alpha:.6f}".format(Alpha=alpha))

            # save weights
            self.model = self.w_vertex_o1
        else:
            progress = tqdm(range(sample_times))
            for p in progress:
                # sample v1, v2
                v1 = self.__sample_src__()
                v2 = self.__sample_tar__(v1)

                # update
                ## positive
                f = np.dot(self.w_vertex[v1], self.w_context[v2])
                f = 1 / (1 + np.exp(-f))
                g = (1 - f) * alpha
                update_value = np.dot(self.w_vertex[v1], g)
                total_update_value = update_value
                self.w_context[v2] = np.add(self.w_context[v2], update_value)
                ## negative
                for _ in range(negative_samples):
                    v2 = self.__sample_neg__()
                    f = np.dot(self.w_vertex[v1], self.w_context[v2])
                    f = 1 / (1 + np.exp(-f))
                    g = (0 - f) * alpha
                    update_value = np.dot(self.w_vertex[v1], g)
                    total_update_value =  np.add(total_update_value, update_value)
                    self.w_context[v2] = np.add(self.w_context[v2], update_value)
                self.w_vertex_o1[v1] = np.add(self.w_vertex_o1[v1], total_update_value)

                # loss
                alpha = alpha * (1 - (p+1)/sample_times)
                progress.set_description("Alpha: {Alpha:.6f}".format(Alpha=alpha))

            # save weights
            self.model = self.w_context

    def predict(self, topk):
        logger.info("Recommend")
        f = open("recommend.txt", "w")
        f.write("")
        f.close()

        for usr in tqdm(self.vid_to_usr):
            candidates = []
            for itm in self.vid_to_itm:
                candidates.append({
                    "item_id": self.vid_to_itm[itm],
                    "score": np.dot(self.model[usr], self.model[itm])
                })
            candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

            f = open("recommend.txt", "a")
            for k, c in  enumerate(candidates[:topk]):
                f.write(f"{self.vid_to_usr[usr]}::{c['item_id']}::{k+1}\n")
            f.close()

        
line = LINE(100, 1)
line.train(1000, 100, 0.01)
line.predict(10)