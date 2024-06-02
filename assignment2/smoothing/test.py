import unittest
import numpy as np
from mathutils import Matrix, Vector
from .explicit_laplace_smoothing import (
    iterative_explicit_laplace_smooth,
    build_combinatorial_laplacian,
)
from data import primitives, meshes
import unittest
import numpy as np
from mathutils import Matrix, Vector
from .explicit_laplace_smoothing import (
    iterative_explicit_laplace_smooth,
    build_combinatorial_laplacian,
)
from data import primitives, meshes


class TestExplicitLaplaceSmoothing(unittest.TestCase):

    def test_build_combinatorial_laplacian_cube(self):
        mesh = primitives.cube()
        L = build_combinatorial_laplacian(mesh)
        self.assertEqual(L.shape, (8, 8))
        self.assertTrue(L[0, 0] == 1)
        self.assertAlmostEqual(L[0, 1], -1 / 3)
