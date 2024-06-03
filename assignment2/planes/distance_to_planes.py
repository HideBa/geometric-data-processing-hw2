from functools import cache

import numpy.random
import numpy as np
from mathutils import Vector, Matrix


class SquaredDistanceToPlanesSolver(object):
    """
    A solver type for computing and minimizing the sum of squared distances to a set of planes.
    """

    # !!! This function will be used for automatic grading, don't edit the signature !!!
    def __init__(self, planes: list[tuple[Vector, Vector]]):
        """
        Prepares the solver to perform squared-distances-to-planes calculations for a given set of planes.

        In order for the distance() and optimal_point() methods to run in constant time,
        pre-processing must be done in __init__()!

        :param planes: The set of planes to use in future calculations.
                       Each plane is represented as a tuple of a point and a normal.
                       The input for a trio of planes would be:
                       ```
                          [(q_0, n_0), (q_1, n_1), (q_2, n_2)]
                       ```
                       Where `q_i` and `n_i` are the point and the normal for plane_i, respectively.
        """

        # HINT: You'll want to save some precomputed results for best performance.
        #       Saving the list of planes directly and iterating over them in your distance() method will work,
        #       but it won't get full points.
        self.planes = planes
        # Precompute matrix
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for q, n in self.planes:
            n = np.array(n) / np.linalg.norm(n)  # Normalize the normal vector
            q = np.array(q)
            A += np.outer(n, n)
            b += np.dot(q, n) * n
        self.A = A
        self.b = b

        self.A2 = np.array([plane[1] for plane in planes])
        self.b2 = np.array([np.dot(plane[0], plane[1]) for plane in planes])

    # !!! This function will be used for automatic grading, don't edit the signature !!!
    def sum_of_squared_distances(self, point: Vector) -> float:
        """
        Computes the sum of squared distances between a given point and each plane.

        If `distance(point, plane_i)` gives the distance between the point and the nearest point on plane_i,
        then this function should be equivalent to:
        ```
            sum(distance(point, plane)**2 for plane in planes)
        ```

        :param point: The point to find distance for.
        :return: The sum of squared distances between the point and all planes, as a float.
        """
        if len(self.planes) <= 0:
            return 0
        # numpy isn't strictly necessary here, but its features can make things easier.
        p = numpy.array(point)

        Ax = np.dot(self.A2, p)
        diff = Ax - self.b2
        row_norms = np.linalg.norm(self.A2, axis=1)
        squared_dist = (diff / row_norms) ** 2
        sum_dist = np.sum(squared_dist)

        return sum_dist

    # !!! This function will be used for automatic grading, don't edit the signature !!!
    def optimal_point(self) -> Vector:
        """
        Finds a point which minimizes the sum of squared distances to all planes.

        This function is not always deterministic!
        For example, with two (non-parallel) planes any point along the line defined by their intersection is optimal.
        The important thing is that the point returned corresponds to the smallest possible sum of squared distances.
        i.e. `solver.distance(solver.optimal_point())` <= `solver.distance({any other point})`.

        :return: A point which minimizes the sum of squared distances.
        """

        if len(self.planes) <= 0:
            return Vector((0, 0, 0))

        # HINT: numpy.linalg.solve() will come in handy here!

        if (
            len(self.planes) == 1
        ):  # it fails to solve because of singular matrices
            return self.planes[0][
                0
            ]  # Just return the point of plane since any points on the plane can be optimal

        if len(self.planes) == 2:
            p1, n1 = np.array(self.planes[0])
            p2, n2 = np.array(self.planes[1])
            n1 = n1 / np.linalg.norm(n1)
            n2 = n2 / np.linalg.norm(n2)

            d = np.cross(n1, n2)

            A = np.array([n1, n2, d])
            b = np.array([np.dot(n1, p1), np.dot(n2, p2), 0])
            p = np.linalg.lstsq(A.T, b, rcond=None)[0]
            return Vector(p)

        # Solve the linear system A * p = b
        p = np.linalg.solve(self.A, self.b)
        return Vector(p)
