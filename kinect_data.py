# -*- coding:utf-8 -*-
# 数据加载器
import numpy as np
import random
import json
import os

class KinectDataLoader:
    def __init__(self, path='./Kinect_train/'):
        self.x_batches = []
        self.y_batches = []
        self.step_size = 200
        for i in os.walk(path):
            for j in i[2]:
                with open(i[0] + '/' + j, 'r') as f:
                    content = ''.join(f.readlines())
                    ls = json.loads(content)
                    temp_batch_x = []
                    k = 0
                    while k < self.step_size:
                        if k < len(ls):
                            temp_batch_x.append([ls[k][key] for key in ls[k]])
                        else:
                            temp_batch_x.append([ls[len(ls) - 1][key] for key in ls[len(ls) - 1]])
                        k += 1
                    self.x_batches.append(np.array(temp_batch_x))
                    self.y_batches.append(np.array(
                        {
                            'right_hand': [1, 0, 0, 0, 0, 0],
                            'left_hand': [0, 1, 0, 0, 0, 0],
                            'right_foot': [0, 0, 1, 0, 0, 0],
                            'left_foot': [0, 0, 0, 1, 0, 0],
                            'defence': [0, 0, 0, 0, 1, 0],
                            'idle': [0, 0, 0, 0, 0, 1]
                        }[i[0].split('/')[2]]))
    
    def next_batch(self, batch_size):
        x = []
        y = []
        for _ in range(batch_size):
            ri = random.randint(0, len(self.x_batches) - 1)
            x.append(self.x_batches[ri])
            y.append(self.y_batches[ri])
        return (np.array(x), np.array(y))
    
    def all_batches(self, each_size):
        finished = False
        counter = 0
        while not finished:
            x = []
            y = []
            for _ in range(each_size):
                x.append(self.x_batches[counter])
                y.append(self.y_batches[counter])
                temp = counter + 1
                counter = temp % len(self.x_batches)
                if temp == counter:
                    finished = True
            yield (np.array(x), np.array(y))