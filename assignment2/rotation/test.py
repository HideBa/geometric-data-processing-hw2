import math
import random
import unittest

import numpy as np
from mathutils import Matrix, Vector
from .axis_of_rotation import (
    rotation_component,
    axis_of_rotation,
    angle_of_rotation,
)
from data import primitives, meshes


class TestRotation(unittest.TestCase):

    def test_rotation_component(self):
        # Test case 1: identity matrix
        transformation = Matrix.Identity(4)
        expected_rotation = np.eye(3)
        self.assertTrue(
            np.allclose(rotation_component(transformation), expected_rotation)
        )

        # Test caes2: translation
        rotation_matrix = Matrix.Rotation(0.5, 3, "X").to_4x4()
        transformation = (
            Matrix.Translation((1, 2, 3)).to_4x4() @ rotation_matrix
        )
        expected_rotation = rotation_matrix.to_3x3()

        self.assertTrue(
            np.allclose(rotation_component(transformation), expected_rotation)
        )

        # Test case 3: Scaling matrix
        scaling_matrix = Matrix.Scale(2, 4).to_4x4()
        transformation = (
            Matrix.Translation((1, 2, 3)).to_4x4() @ scaling_matrix
        )
        expected_rotation = np.eye(3)
        self.assertTrue(
            np.allclose(rotation_component(transformation), expected_rotation)
        )

        # Test case 4: Combined transformation matrix
        rotation_matrix = Matrix.Rotation(0.5, 3, "X").to_4x4()
        scaling_matrix = Matrix.Scale(2, 4).to_4x4()
        transformation = (
            Matrix.Translation((1, 2, 3)).to_4x4()
            @ scaling_matrix
            @ rotation_matrix
        )
        expected_rotation = rotation_matrix.to_3x3()
        self.assertTrue(
            np.allclose(rotation_component(transformation), expected_rotation)
        )
