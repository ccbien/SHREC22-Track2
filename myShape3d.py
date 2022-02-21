import numpy as np
import random

def get_rotationMatrix_from_vectors(u, v):
    """
    Create a rotation matrix that rotates the space from a 3D vector `u` to a 3D vector `v`

    :param u: Orign vector `np.array (1,3)`.
    :param v: Destiny vector `np.array (1,3)`.

    :returns: Rotation matrix `np.array (3, 3)`

    ---
    """

    # Lets find a vector which is ortogonal to both u and v
    w = np.cross(u, v)

    # This orthogonal vector w has some interesting proprieties
    # |w| = sin of the required rotation
    # dot product of w and goal_normal_plane is the cos of the angle
    c = np.dot(u, v)
    s = np.linalg.norm(w)

    # Now, we compute rotation matrix from rodrigues formula
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    # We calculate the skew symetric matrix of the ort_vec
    Sx = np.asarray([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = np.eye(3) + Sx + Sx.dot(Sx) * ((1 - c) / (s ** 2))
    return R


def rodrigues_rot(P, n0, n1):
    """
    Rotate a set of point between two normal vectors using Rodrigues' formula.

    :param P: Set of points `np.array (N,3)`.
    :param n0: Orign vector `np.array (1,3)`.
    :param n1: Destiny vector `np.array (1,3)`.

    :returns: Set of points P, but rotated `np.array (N, 3)`

    ---
    """

    # If P is only 1d array (coords of single point), fix it to be matrix
    P = np.asarray(P)
    if P.ndim == 1:
        P = P[np.newaxis, :]

    # Get vector of rotation k and angle theta
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    k = np.cross(n0, n1)
    P_rot = np.zeros((len(P), 3))
    if np.linalg.norm(k) != 0:
        k = k / np.linalg.norm(k)
        theta = np.arccos(np.dot(n0, n1))

        # Compute rotated points
        for i in range(len(P)):
            P_rot[i] = (
                P[i] * np.cos(theta) + np.cross(k, P[i]) * np.sin(theta) + k * np.dot(k, P[i]) * (1 - np.cos(theta))
            )
    else:
        P_rot = P
    return P_rot

class Plane:
    def __init__(self):
        self.loss = 0
        self.center = np.array([])
        self.inliers = np.array([])
        self.normal = np.array([])

    def fit(self, pts, thresh=0.1):
        # Find PCA plane
        # Normal and center
        covariance_matrix = np.cov(pts.T)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        id = np.argmin(eigen_values)
        self.normal = eigen_vectors[:,id] 
        self.normal = self.normal / np.linalg.norm(self.normal)

        self.center = np.mean(pts, axis=0)


        # Equation: vec[0]*x + vec[1]*y + vec[2]*z = -D
        D = -np.sum(np.multiply(self.normal, self.center))
        equation = np.array([self.normal[0], self.normal[1], self.normal[2], D])

        dist_pt = (
            equation[0] * pts[:, 0] + equation[1] * pts[:, 1] + equation[2] * pts[:, 2] + equation[3]
        ) /  np.sqrt(equation[0] ** 2 + equation[1] ** 2 + equation[2] ** 2)
        
        thresh = np.max(dist_pt)*0.1
        self.inliers = np.where(np.abs(dist_pt) <= thresh)[0]

        self.loss = np.mean(np.abs(dist_pt))


        return self.normal, self.center, self.loss, self.inliers


class Sphere:
    def __init__(self):
        self.inliers = []
        self.center = []
        self.radius = 0

    def fit(self, pts, p=0.05, k=0.3, maxIteration=100):
        
        n_points = pts.shape[0]
        best_inliers = self.inliers

        for it in range(maxIteration):

            # Samples 4 random points
            N = int(k*n_points)
            id_samples = random.sample(range(0, n_points), N)
            pt_samples = pts[id_samples]

            spX = pt_samples[:,0]
            spY = pt_samples[:,1]
            spZ = pt_samples[:,2]
            A = np.zeros((len(spX), 4))
            A[:,0] = spX*2
            A[:,1] = spY*2
            A[:,2] = spZ*2
            A[:,3] = 1

            
            #   Assemble the f matrix
            f = np.zeros((len(spX),1))
            f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
            
            # Now we calculate the center and radius
            C, residules, rank, singval = np.linalg.lstsq(A,f)
            center = np.array([C[0,0], C[1,0], C[2,0]])
            t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
            radius = np.sqrt(t)[0]

            # Distance from a point
            pt_id_inliers = []  # list of inliers ids
            dist_pt = center - pts
            dist_pt = np.linalg.norm(dist_pt, axis=1)

            # Select indexes where distance is biggers than the threshold
            thresh = p*radius
            pt_id_inliers = np.where(np.abs(dist_pt - radius) <= thresh)[0]

            if len(pt_id_inliers) > len(best_inliers):
                best_inliers = pt_id_inliers
                self.inliers = best_inliers
                self.center = center
                self.radius = radius


        return self.center, self.radius, self.inliers
