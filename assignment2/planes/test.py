import unittest
import numpy as np
import numpy.random
from mathutils import Matrix, Vector
from .distance_to_planes import SquaredDistanceToPlanesSolver


class TestDistanceToPlanes(unittest.TestCase):

    # HINT: Add your own unit tests here

    # HINT: CreateExamplePlanesOperator in __init__.py contains code which creates a 'cube' of planes,
    #       This could be adapted into a nice test case, because the optimal point will always be in the middle.

    def test_one_plane(self):
        # Create a horizontal plane which is the same as the XY plane
        plane = (Vector([0, 0, 0]), Vector([0, 0, 1]))

        # Create a solver for distances from this plane
        solver = SquaredDistanceToPlanesSolver([plane])

        # Any point's distance to the plane should be equal to its height squared
        for _ in range(100):
            point = Vector(numpy.random.uniform(-100, 100, 3))
            dist = solver.sum_of_squared_distances(point)
            # print("distance :", dist)
            self.assertEqual(
                dist,
                point.z**2,
                "Distance squared from the XY plane should be equal to z^2",
            )

    def test_empty_planes(self):
        solver = SquaredDistanceToPlanesSolver([])
        self.assertEqual(
            solver.sum_of_squared_distances(Vector((0, 0, 0))), 0.0
        )
        self.assertEqual(solver.optimal_point(), Vector((0, 0, 0)))

    def test_single_plane(self):
        solver = SquaredDistanceToPlanesSolver(
            [(Vector((0, 0, 0)), Vector((1, 0, 0)))]
        )
        self.assertEqual(
            solver.sum_of_squared_distances(Vector((1, 1, 0))), 1.0
        )
        self.assertEqual(solver.optimal_point(), Vector((0, 0, 0)))

    def test_two_planes(self):
        solver = SquaredDistanceToPlanesSolver(
            [
                (Vector((0, 0, 0)), Vector((1, 0, 0))),
                (Vector((0, 0, 0)), Vector((0, 1, 0))),
            ]
        )
        self.assertEqual(
            solver.sum_of_squared_distances(Vector((1, 1, 0))), 2.0
        )
        self.assertEqual(solver.optimal_point(), Vector((0, 0, 0)))

    def test_three_planes(self):
        solver = SquaredDistanceToPlanesSolver(
            [
                (Vector((0, 0, 0)), Vector((1, 0, 0))),
                (Vector((0, 0, 0)), Vector((0, 1, 0))),
                (Vector((0, 0, 0)), Vector((0, 0, 1))),
            ]
        )
        self.assertEqual(
            solver.sum_of_squared_distances(Vector((1, 1, 1))), 3.0
        )
        self.assertEqual(solver.optimal_point(), Vector((0, 0, 0)))

    def test_cubic(self):
        solver = SquaredDistanceToPlanesSolver(
            [
                (Vector((0, 0, 0)), Vector((1, 0, 0))),
                (Vector((0, 0, 0)), Vector((0, 1, 0))),
                (Vector((0, 0, 0)), Vector((0, 0, 1))),
                (Vector((1, 1, 1)), Vector((-1, 0, 0))),
                (Vector((1, 1, 1)), Vector((0, -1, 0))),
                (Vector((1, 1, 1)), Vector((0, 0, -1))),
            ]
        )
        self.assertEqual(
            solver.sum_of_squared_distances(Vector((0.5, 0.5, 0.5))),
            0.25 * 6,
        )
        self.assertEqual(solver.optimal_point(), Vector((0.5, 0.5, 0.5)))

        solver = SquaredDistanceToPlanesSolver(
            [
                (Vector((0, 0, 0)), Vector((100, 0, 0))),
                (Vector((0, 0, 0)), Vector((0, 50, 0))),
                (Vector((0, 0, 0)), Vector((0, 0, 10))),
                (Vector((2, 2, 2)), Vector((-10, 0, 0))),
                (Vector((2, 2, 2)), Vector((0, -30, 0))),
                (Vector((2, 2, 2)), Vector((0, 0, -50))),
            ]
        )
        self.assertEqual(
            solver.sum_of_squared_distances(Vector((1, 1, 1))),
            6,
        )
        self.assertEqual(
            solver.optimal_point(),
            Vector(
                (
                    1,
                    1,
                    1,
                )
            ),
        )
