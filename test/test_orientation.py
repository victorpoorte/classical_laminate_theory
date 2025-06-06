import math
import unittest
from test.test_strategies import my_list_of_numbers_strategy

import hypothesis.strategies as st
import numpy as np
from hypothesis import example, given

from clt.orientation import Orientation


class MyOnlineExampleStrain:
    # Online example can be found below
    # https://www.youtube.com/watch?v=sI3S2ytiHnM

    @property
    def strain_vector(self):
        return np.array([0.003453, -0.004671, 0])

    @property
    def expected_local_strain(self):
        return np.array([0.001422, -0.002640, -0.007036])


class MyOnlineExampleTestOrientation(Orientation):
    # Online example can be found below
    # https://www.youtube.com/watch?v=sI3S2ytiHnM

    @property
    def expected_rotation_matrix(self):
        return np.array(
            [
                [0.750, 0.250, 0.433],
                [0.250, 0.750, -0.433],
                [-0.866, 0.866, 0.500],
            ]
        )

    @property
    def expected_T_2_rotation_matrix(self):
        return np.array(
            [
                [0.750, 0.250, -0.433],
                [0.250, 0.750, 0.433],
                [0.866, -0.866, 0.500],
            ]
        )

    @classmethod
    def instantiate(cls):
        return cls(30, check_rotation=False)


@st.composite
def my_orientation_strategy(draw):
    orientation = draw(st.decimals(min_value=0, max_value=90))

    return Orientation(float(orientation), check_rotation=False)


@st.composite
def my_strain_strategy(draw):
    strain_entries = 3

    return np.array(draw(my_list_of_numbers_strategy(strain_entries)))


class TestOrientation(unittest.TestCase):
    def test_init(self) -> None:
        rotation = 45
        orientation = Orientation(rotation)
        expected_value = math.pi / 4
        actual_value = orientation.rotation
        self.assertEqual(expected_value, actual_value)

        rotation = math.pi / 6
        orientation = Orientation(rotation, degree=False)
        actual_value = orientation.rotation
        self.assertEqual(rotation, actual_value)

    def test_is_valid_rotation(self):
        rotation = 0.1
        with self.assertRaises(ValueError) as context:
            Orientation(rotation)

        self.assertEqual(
            str(context.exception),
            f"Value of rotation ({rotation}) seems invalid."
            + "Set 'check_rotation' to false to bypass this error.",
        )

    @given(my_orientation_strategy())
    def test_m(self, orientation: Orientation) -> None:
        self.assertEqual(orientation.m, math.cos(orientation.rotation))

    @given(my_orientation_strategy())
    def test_n(self, orientation: Orientation) -> None:
        self.assertEqual(orientation.n, math.sin(orientation.rotation))

    @given(my_orientation_strategy())
    def test_rotation_matrix(self, orientation: Orientation) -> None:
        expected_value = np.array(
            [
                [
                    orientation.m**4,
                    orientation.n**4,
                    2 * orientation.m**2 * orientation.n**2,
                    4 * orientation.m**2 * orientation.n**2,
                ],
                [
                    orientation.n**4,
                    orientation.m**4,
                    2 * orientation.m**2 * orientation.n**2,
                    4 * orientation.m**2 * orientation.n**2,
                ],
                [
                    orientation.m**2 * orientation.n**2,
                    orientation.m**2 * orientation.n**2,
                    -2 * orientation.m**2 * orientation.n**2,
                    (orientation.m**2 - orientation.n**2) ** 2,
                ],
                [
                    orientation.m**2 * orientation.n**2,
                    orientation.m**2 * orientation.n**2,
                    orientation.m**4 + orientation.n**4,
                    -4 * orientation.m**2 * orientation.n**2,
                ],
                [
                    orientation.m**3 * orientation.n,
                    -orientation.m * orientation.n**3,
                    orientation.m * orientation.n**3
                    - orientation.m**3 * orientation.n,
                    2
                    * (
                        orientation.m * orientation.n**3
                        - orientation.m**3 * orientation.n
                    ),
                ],
                [
                    orientation.m * orientation.n**3,
                    -orientation.m**3 * orientation.n,
                    orientation.m**3 * orientation.n
                    - orientation.m * orientation.n**3,
                    2
                    * (
                        orientation.m**3 * orientation.n
                        - orientation.m * orientation.n**3
                    ),
                ],
            ]
        )
        np.testing.assert_array_equal(
            expected_value, orientation.rotation_matrix
        )

    @given(my_orientation_strategy())
    @example(MyOnlineExampleTestOrientation.instantiate())
    def test_strain_rotation_matrix_2D(self, orientation: Orientation) -> None:
        m, n = orientation.m, orientation.n
        expected_value = np.array(
            [
                [m**2, n**2, m * n],
                [n**2, m**2, -m * n],
                [-2 * m * n, 2 * m * n, m**2 - n**2],
            ]
        )

        if orientation == MyOnlineExampleTestOrientation.instantiate():
            expected_value = (
                MyOnlineExampleTestOrientation.instantiate().expected_rotation_matrix
            )

        actual_value = orientation.strain_rotation_matrix_2D

        np.testing.assert_array_almost_equal(
            expected_value, actual_value, decimal=3
        )

    @given(my_orientation_strategy())
    @example(MyOnlineExampleTestOrientation.instantiate())
    def test_T_2_rotation_matrix(self, orientation: Orientation) -> None:
        m, n = orientation.m, orientation.n
        expected_value = np.array(
            [
                [m**2, n**2, -m * n],
                [n**2, m**2, m * n],
                [2 * m * n, -2 * m * n, m**2 - n**2],
            ]
        )

        if orientation == MyOnlineExampleTestOrientation.instantiate():
            expected_value = (
                MyOnlineExampleTestOrientation.instantiate().expected_T_2_rotation_matrix
            )

        actual_value = orientation.T_2_rotation_matrix

        np.testing.assert_array_almost_equal(
            expected_value, actual_value, decimal=3
        )

    def test_strain_rotation_matrix_3D(self):
        orientation = Orientation(0)
        expected_value = np.zeros((6, 6))
        expected_value[0, 0] = 1
        expected_value[1, 1] = 1
        expected_value[2, 2] = 1
        expected_value[3, 3] = 1
        expected_value[4, 4] = 1
        expected_value[5, 5] = 1
        actual_value = orientation.strain_rotation_matrix_3D
        np.testing.assert_array_equal(expected_value, actual_value)

        orientation = Orientation(90)
        expected_value = np.zeros((6, 6))
        expected_value[1, 0] = 1
        expected_value[0, 1] = 1
        expected_value[2, 2] = 1
        expected_value[3, 4] = -1
        expected_value[4, 3] = 1
        expected_value[5, 5] = -1
        actual_value = orientation.strain_rotation_matrix_3D
        np.testing.assert_array_almost_equal(expected_value, actual_value)

        var = 1 / math.sqrt(2)
        orientation = Orientation(45)
        expected_value = np.zeros((6, 6))
        expected_value[0, 0] = var**2
        expected_value[1, 1] = var**2
        expected_value[2, 2] = 1
        expected_value[3, 3] = var
        expected_value[4, 4] = var
        expected_value[0, 1] = var**2
        expected_value[1, 0] = var**2
        expected_value[0, -1] = var**2
        expected_value[1, -1] = -(var**2)
        expected_value[-1, 0] = -2 * var**2
        expected_value[-1, 1] = 2 * var**2
        expected_value[3, 4] = -var
        expected_value[4, 3] = var
        actual_value = orientation.strain_rotation_matrix_3D
        np.testing.assert_array_almost_equal(expected_value, actual_value)

    def test_stress_rotation_matrix_3D(self):
        orientation = Orientation(0)
        expected_value = np.zeros((6, 6))
        expected_value[0, 0] = 1
        expected_value[1, 1] = 1
        expected_value[2, 2] = 1
        expected_value[3, 3] = 1
        expected_value[4, 4] = 1
        expected_value[5, 5] = 1
        actual_value = orientation.stress_rotation_matrix_3D
        np.testing.assert_array_equal(expected_value, actual_value)

        orientation = Orientation(90)
        expected_value = np.zeros((6, 6))
        expected_value[1, 0] = 1
        expected_value[0, 1] = 1
        expected_value[2, 2] = 1
        expected_value[3, 4] = -1
        expected_value[4, 3] = 1
        expected_value[5, 5] = -1
        actual_value = orientation.stress_rotation_matrix_3D
        np.testing.assert_array_almost_equal(expected_value, actual_value)

        var = 1 / math.sqrt(2)
        orientation = Orientation(45)
        expected_value = np.zeros((6, 6))
        expected_value[0, 0] = var**2
        expected_value[1, 1] = var**2
        expected_value[2, 2] = 1
        expected_value[3, 3] = var
        expected_value[4, 4] = var
        expected_value[0, 1] = var**2
        expected_value[1, 0] = var**2
        expected_value[0, -1] = 2 * var**2
        expected_value[1, -1] = -2 * var**2
        expected_value[-1, 0] = -(var**2)
        expected_value[-1, 1] = var**2
        expected_value[3, 4] = -var
        expected_value[4, 3] = var
        actual_value = orientation.stress_rotation_matrix_3D
        np.testing.assert_array_almost_equal(expected_value, actual_value)

    @given(my_orientation_strategy(), my_strain_strategy())
    @example(
        MyOnlineExampleTestOrientation.instantiate(),
        MyOnlineExampleStrain().strain_vector,
    )
    def test_compute_local_strains(
        self, orientation: Orientation, strains_vector: np.ndarray
    ) -> None:
        expected_value = np.dot(
            orientation.strain_rotation_matrix_2D, strains_vector
        )
        if (
            orientation == MyOnlineExampleTestOrientation.instantiate()
            and np.array_equal(
                strains_vector, MyOnlineExampleStrain().strain_vector
            )
        ):
            expected_value = MyOnlineExampleStrain().expected_local_strain

        actual_value = orientation.compute_local_strains(strains_vector)

        np.testing.assert_array_almost_equal(expected_value, actual_value)

    def test_rotation_degree(self):
        rotation = 45
        orientation = Orientation(rotation)
        expected_value = rotation
        actual_value = orientation.rotation_degree
        self.assertEqual(expected_value, actual_value)

    def test_get_rotation_matrix(self):
        rotation = 45
        orientation = Orientation(rotation)

        strain_size = 3
        strains = np.zeros((strain_size,))
        rotation_matrix = orientation._get_rotation_matrix(strains)
        actual_value = rotation_matrix.shape
        expected_value = (strain_size, strain_size)
        self.assertEqual(expected_value, actual_value)

        strain_size = 6
        strains = np.zeros((strain_size,))
        rotation_matrix = orientation._get_rotation_matrix(strains)
        actual_value = rotation_matrix.shape
        expected_value = (strain_size, strain_size)
        self.assertEqual(expected_value, actual_value)

        with self.assertRaises(ValueError) as cm:
            orientation._get_rotation_matrix(np.zeros((9,)))
        self.assertEqual(str(cm.exception), "Invalid strain vector shape...")

    def test_is_90_degrees(self):
        self.assertFalse(Orientation(45).is_90_degrees)
        self.assertTrue(Orientation(90).is_90_degrees)


if __name__ == "__main__":
    unittest.main()


# End
