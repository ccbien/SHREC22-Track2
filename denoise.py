import numpy as np
import open3d as o3d
import numpy as np

def denoise_pcd(pcd, N=50, epsilon=0.1, p=0.1):
    # filtering multiple times will reduce the noise significantly
    # but may cause the points distribute unevenly on the surface.
    guided_filter(pcd, N, epsilon)
    #guided_filter(pcd, 0.01, 0.1)

    #o3d.visualization.draw_geometries([pcd])


def guided_filter(pcd, N=50, epsilon=0.1, p=0.1):
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    points_copy = np.array(pcd.points)
    points = np.asarray(pcd.points)
    num_points = len(pcd.points)
    
    for i in range(num_points):
        N = min(N, int(0.1*num_points))
        k, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], N)
        if k < 3:
            continue

        neighbors = points[idx, :]
        mean = np.mean(neighbors, 0)
        cov = np.cov(neighbors.T)
        e = np.linalg.inv(cov + epsilon * np.eye(3))

        A = cov @ e
        b = mean - A @ mean

        points_copy[i] = A @ points[i] + b

    pcd.points = o3d.utility.Vector3dVector(points_copy)

