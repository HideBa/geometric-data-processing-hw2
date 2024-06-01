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

    def test_axis_of_rotation(self):

        angle = 0.5
        transformation = Matrix.Rotation(angle, 3, "X")
        expected_axis = Vector((1, 0, 0))
        self.assertEqual(axis_of_rotation(transformation), expected_axis)

        transformation = Matrix.Rotation(angle, 3, "Y")
        expected_axis = transformation.to_quaternion().axis
        self.assertEqual(axis_of_rotation(transformation), expected_axis)

        transformation = Matrix.Rotation(angle, 3, "Z")
        expected_axis = transformation.to_quaternion().axis
        self.assertEqual(axis_of_rotation(transformation), expected_axis)

        angle_x = 0.5
        angle_y = 0.3
        angle_z = 0.7
        transformation = (
            Matrix.Rotation(angle_x, 3, "X")
            @ Matrix.Rotation(angle_y, 3, "Y")
            @ Matrix.Rotation(angle_z, 3, "Z")
        )
        expected_axis = transformation.to_quaternion().axis
        self.assertTrue(
            np.allclose(axis_of_rotation(transformation), expected_axis)
        )

    def test_angle_of_rotation(self):
        # Identity matrix
        transformation = Matrix.Identity(3)
        expected_angle = 0
        self.assertEqual(angle_of_rotation(transformation), expected_angle)

        # Rotation around individual axes
        angle_x = 0.5
        angle_y = 0.3
        angle_z = 0.7
        transformations = [
            Matrix.Rotation(angle_x, 3, "X"),
            Matrix.Rotation(angle_y, 3, "Y"),
            Matrix.Rotation(angle_z, 3, "Z"),
        ]
        expected_angles = [angle_x, angle_y, angle_z]
        for transformation, expected_angle in zip(
            transformations, expected_angles
        ):
            self.assertAlmostEqual(
                angle_of_rotation(transformation), expected_angle, 5
            )

        # Combined rotations
        combined_transformation = (
            Matrix.Rotation(angle_x, 3, "X")
            @ Matrix.Rotation(angle_y, 3, "Y")
            @ Matrix.Rotation(angle_z, 3, "Z")
        )
        expected_angle = math.acos(
            (
                combined_transformation[0][0]
                + combined_transformation[1][1]
                + combined_transformation[2][2]
                - 1
            )
            / 2
        )
        self.assertAlmostEqual(
            angle_of_rotation(combined_transformation),
            expected_angle,
            places=5,
        )
