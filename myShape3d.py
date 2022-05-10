import random
import numpy as np

class Plane:
    def __init__(self):
        self.loss = 0
        self.center = []
        self.inliers = []
        self.normal = []

    def fit(self, pc, eps=0.1):
        pts = np.asarray(pc.points)
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
        
        thresh = np.max(dist_pt)*eps
        self.inliers = np.where(np.abs(dist_pt) <= thresh)[0]

        self.loss = np.mean(np.abs(dist_pt))


        return self.normal, self.center, self.loss, self.inliers


class Sphere:
    def __init__(self):
        self.inliers = []
        self.center = []
        self.radius = 0

    def fit(self, pc, eps=0.05, sample_ratio=0.3, maxIteration=100):
        pts = np.asarray(pc.points)
        n_points = pts.shape[0]
        best_inliers = self.inliers

        for it in range(maxIteration):

            # Samples random points
            N = int(sample_ratio*n_points)
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
            C, residules, rank, singval = np.linalg.lstsq(A, f, rcond=None)
            center = np.array([C[0,0], C[1,0], C[2,0]])
            t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
            radius = np.sqrt(t)[0]

            # Distance from a point
            pt_id_inliers = []  # list of inliers ids
            dist_pt = center - pts
            dist_pt = np.linalg.norm(dist_pt, axis=1)

            # Select indexes where distance is biggers than the threshold
            thresh = eps*radius
            pt_id_inliers = np.where(np.abs(dist_pt - radius) <= thresh)[0]

            if len(pt_id_inliers) > len(best_inliers):
                best_inliers = pt_id_inliers
                self.inliers = best_inliers
                self.center = center
                self.radius = radius


        return self.center, self.radius, self.inliers


class Cone:
    def __init__(self):
        self.apex = []
        self.axis_vector = []
        self.theta = 0
        self.mean_dist = 0
    
    @staticmethod
    def get_angle(u, v):
        t = np.linalg.norm(u) * np.linalg.norm(v)
        return np.abs(np.arccos(np.dot(u, v) / t))

    @staticmethod
    def calculate_distances_to_cone(apex, axis_vector, theta, points):
        U = points - apex
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        v = axis_vector / np.linalg.norm(axis_vector)

        angles = np.abs(np.arccos(np.dot(U, v)))
        angle_errors = np.abs(angles - theta)
        mask = angle_errors > (np.pi / 2)

        distances_to_apex = np.sqrt(np.sum((points - apex)**2, axis=1))
        res = distances_to_apex * np.sin(angle_errors)
        res = mask * distances_to_apex + (1 - mask) * res
        return res

    @staticmethod
    def calculate_ratio(apex, axis_vector, theta, points, eps=1e-3):
        ds = Cone.calculate_distances_to_cone(apex, axis_vector, theta, points)
        return np.sum(ds < eps) / ds.shape[0]

    @staticmethod
    def calculate_cone_params_simple(points, normals):
        # Apex
        B = np.zeros(3)
        B[:] = normals[:,0] * points[:,0] + normals[:,1] * points[:,1] + normals[:,2] * points[:,2]
        try:
            apex = np.linalg.solve(normals, B) 
        except:
            return None, None, None

        # axis normal
        P = [None, None, None]
        for i in range(3):
            P[i] = points[i] - apex
            P[i] /= np.linalg.norm(P[i])
        u = P[1] - P[0]
        v = P[2] - P[0]
        axis_vector = np.cross(u, v)
        axis_vector /= np.linalg.norm(axis_vector)

        # half the aperture - theta
        theta = Cone.get_angle(points[0] - apex, axis_vector)
        if theta > np.pi / 2:
            return None, None, None

        return apex, axis_vector, theta

    def fit(self, pc, eps=1e-3, sample_ratio=1.0, min_points_count=1e9, maxIteration=1000):
        pc.estimate_normals()
        pc.normalize_normals()
        all_points = np.asarray(pc.points)
        all_normals = np.asarray(pc.normals)
        
        mn = all_points.min(axis=0)
        mx = all_points.max(axis=0)
        L = np.sqrt(np.sum((mx - mn)**2))
        
        
        best_ratio = 0
        # best_mean_dist = np.inf
        best_apex = None
        best_axis_vector = None
        best_theta = None

        if all_points.shape[0] <= min_points_count:
            sampled_points = np.copy(all_points)
        else:
            n_sample = int(max(all_points.shape[0] * sample_ratio, min_points_count))
            indices = random.sample(list(range(all_points.shape[0])), n_sample)
            sampled_points = all_points[indices]

        count = 0
        while count < maxIteration:
            # Calculate params
            indices = np.random.choice(all_points.shape[0], 3, replace=False)
            normals = all_normals[indices] # sample three points, normals
            points = all_points[indices]

            apex, axis_vector, theta = Cone.calculate_cone_params_simple(points, normals)
            if apex is None:
                continue
            
            count += 1
            ratio = Cone.calculate_ratio(apex, axis_vector, theta, sampled_points, eps=eps)
            # mean_dist = calculate_distances_to_cone(apex, axis_vector, theta, sampled_points).mean()
            if best_ratio < ratio:
            # if best_mean_dist > mean_dist:
                best_ratio = ratio
                # best_mean_dist = mean_dist
                best_apex = apex
                best_axis_vector = axis_vector
                best_theta = theta

        return best_apex, best_axis_vector, best_theta, best_ratio