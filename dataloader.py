import os
from os.path import join
import pickle
import random
import numpy as np


def read_txt(path):
    points = []
    with open(path, 'r') as f:
        for line in f.readlines():
            try:
                x, y, z = map(float, line.split(','))
                points.append([x, y, z])
            except:
                pass
    return np.array(points)


def read_ground_truth(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return {
        'label': int(lines[0]),
        'params': list(map(float, lines[1:]))
    }


class TrainLoader:
    def __init__(self, root):
        self.label = {}
        self.paths = {i + 1 : [] for i in range(5)}
        id2label = pickle.load(open('./id2label.pkl', 'rb'))
        for file in os.listdir(root):
            path = join(root, file)
            id = int(file.replace('pointCloud', '').replace('.txt', ''))
            label = id2label[id]
            self.label[path] = label
            self.paths[label].append(path)


    def load_item(self, path, points_only=False):
        if points_only:
            return read_txt(path)
        return read_txt(path), self.label[path]


    def sample(self, N, label=None, points_only=False):
        if label is not None:
            paths = self.paths[label]
        else:
            paths = []
            for lst in self.paths.values:
                paths += lst
        data = []
        for path in random.sample(paths, N):
            data.append(self.load_item(path, points_only=points_only))
        return data