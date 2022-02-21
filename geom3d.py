from abc import ABC, abstractmethod

import numpy as np

from _descriptor import Direction, Position, PositiveNumber


class GeometricShape(ABC):
    @abstractmethod
    def distance_to_point(self, point):
        """Calculates the smallest distance from a point to the shape"""

    # @abstractmethod
    # def project_point(self, point):
    # pass


class Circle3D(GeometricShape):
    center = Position(3)
    direction = Direction(3)
    radius = PositiveNumber()

    def __init__(self, center, direction, radius):
        self.center = center
        self.direction = direction
        self.radius = radius

    def __repr__(self):
        return f"Circle3D(center={self.center.tolist()}, direction={self.direction.tolist()}, radius={self.radius})"

    def distance_to_point(self, point):
        delta_p = point - self.center
        x1 = np.matmul(
            np.expand_dims(np.dot(delta_p, self.direction), axis=-1),
            np.atleast_2d(self.direction),
        )
        x2 = delta_p - x1
        return np.sqrt(
            np.linalg.norm(x1, axis=-1) ** 2
            + (np.linalg.norm(x2, axis=-1) - self.radius) ** 2
        )


class Torus(Circle3D):
    minor_radius = PositiveNumber()

    def __init__(self, center, direction, major_radius, minor_radius):
        super().__init__(center, direction, major_radius)
        self.minor_radius = minor_radius

    def __repr__(self):
        return f"Torus(center={self.center.tolist()}, direction={self.direction.tolist()}, major_radius={self.major_radius}, minor_radius={self.minor_radius})"

    @property
    def major_radius(self):
        return self.radius

    def distance_to_point(self, point):
        return np.abs(super().distance_to_point(point) - self.minor_radius)
