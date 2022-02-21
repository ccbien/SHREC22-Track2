from ast import arg
import random
from matplotlib.pyplot import axes
import numpy as np
import open3d as o3d
from torch import rand

from utils import get_angles


def _arctan(x, y):
    if x > 0 and y > 0:
        return np.arctan(y / x)
    elif x < 0:
        return np.arctan(y / x) + np.pi
    else:
        return np.arctan(y / x) + 2 * np.pi


def honv(normals=None, pcd=None, filepath=None, bins=10, flatten=False, normalize=True):
    if normals is None:
        if pcd is None:
            assert filepath is not None
            pcd = o3d.io.read_point_cloud(filepath)
        if not pcd.has_normals():
            pcd.estimate_normals()
            pcd.normalize_normals()
        normals = np.asarray(pcd.normals)

    Ox = np.array([1, 0, 0])
    Oz = np.array([0, 0, 1])
    projection = np.zeros_like(normals)
    projection[:, 0:2] = normals[:, 0:2]
    phis = get_angles(projection, Ox)
    thetas = get_angles(normals, Oz)

    hist, _, _ = np.histogram2d(phis, thetas, bins=bins, range=[[0, np.pi], [0, np.pi]])
    hist /= hist.sum()

    feat = np.copy(hist)
    if normalize:
        rows, cols = np.where(hist == hist.max())
        row, col = np.unravel_index(hist.argmax(), hist.shape) 
        feat = np.roll(feat, -row + hist.shape[0]//2, axis=0)
        feat = np.roll(feat, -col + hist.shape[1]//2, axis=1)
    if flatten:
        feat = feat.reshape(-1)
    return feat


def surflet_pairs_feature(points=None, normals=None, filepath=None, n_pairs=1000, bins=(5, 5, 5, 5), flatten=True):
    if points is None or normals is None:
        pcd = o3d.io.read_point_cloud(filepath)
        pcd.estimate_normals()
        pcd.normalize_normals()
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

    indices = list(range(points.shape[0]))
    idx1 = np.array([random.choice(indices) for i in range(n_pairs)])
    idx2 = []
    for i in range(n_pairs):
        a, b = random.sample(indices, 2)
        if a != idx1[i]:
            idx2.append(a)
        else:
            idx2.append(b)

    P1 = points[idx1]
    P2 = points[idx2]
    N1 = normals[idx1]
    N2 = normals[idx2]

    f1, f2, f3, f4 = [], [], [], []
    for p1, p2, n1, n2 in zip(P1, P2, N1, N2):
        u = np.copy(n1)
        v = np.cross(p2 - p1, u)
        v /= np.linalg.norm(v)
        w = np.cross(u, v)
        f1.append(_arctan(np.dot(w, n2), np.dot(u, n2))) # 0 --> 2pi
        f2.append(np.dot(v, n2)) # 0 --> 1
        f3.append(np.dot(u, p2 - p1) / np.linalg.norm(p2 - p1)) # 0 --> 1
        f4.append(np.linalg.norm(p2 - p1))

    f1, f2, f3, f4 = map(np.array, (f1, f2, f3, f4))
    f1, f2, f3, f4 = map(np.expand_dims, (f1, f2, f3, f4), (-1, -1, -1, -1))
    f1 /= 2 * np.pi
    f4 /= f4.max()

    hist, _ =  np.histogramdd(
        np.concatenate([f1, f2, f3, f4], axis=1),
        bins=bins
    )
    if flatten:
        hist = hist.reshape(-1)
    hist /= hist.sum()
    return hist


def get_histogram(normals=None, pcd=None, filepath=None, bins=10, flatten=True, normalize=True, max_ratio=1.0):
    if normals is None:
        if pcd is None:
            assert filepath is not None
            pcd = o3d.io.read_point_cloud(filepath)
        if not pcd.has_normals():
            pcd.estimate_normals()
            pcd.normalize_normals()
        normals = np.asarray(pcd.normals)

    Ox = np.array([1, 0, 0])
    Oz = np.array([0, 0, 1])
    projection = np.zeros_like(normals)
    projection[:, 0:2] = normals[:, 0:2]
    phis = get_angles(projection, Ox)
    thetas = get_angles(normals, Oz)

    hist, _, _ = np.histogram2d(phis, thetas, bins=bins, range=[[0, np.pi], [0, np.pi]])
    hist /= hist.sum()

    if normalize:
        result = []
        rows, cols = np.where(hist >= hist.max() * max_ratio)
        for row, col in zip(rows, cols):
            temp = np.copy(hist)
            temp = np.roll(temp, -row + hist.shape[0]//2, axis=0)
            temp = np.roll(temp, -col + hist.shape[1]//2, axis=1)
            result.append(temp)
    else:
        result = [hist]

    if flatten:
        result = [x.reshape(-1) for x in result]

    return result