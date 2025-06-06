import unittest
from test.test_material import (
    MyOnlineExerciseTestMaterial,
    my_lamina_strategy,
    my_strain_strategy,
    my_thick_lamina_strategy,
)
from test.test_orientation import my_orientation_strategy

import hypothesis.strategies as st
import numpy as np
from hypothesis import example, given

from analytical_vessel_analysis.clt.clt.laminate_layer import LaminateLayer
from clt.orientation import Orientation


class MyTestLayer(LaminateLayer):
    thickness = 0.005

    @classmethod
    def layer_30(cls):
        return cls(
            MyOnlineExerciseTestMaterial.instantiate(),
            cls.thickness,
            Orientation(30),
        )

    @classmethod
    def test_layer_negative_30(cls):
        return cls(
            MyOnlineExerciseTestMaterial.instantiate(),
            cls.thickness,
            Orientation(-30),
        )


@st.composite
def my_thin_layer_strategy(draw):
    return LaminateLayer(
        draw(my_lamina_strategy()),
        float(draw(st.decimals(allow_infinity=False, allow_nan=False))),
        draw(my_orientation_strategy()),
    )


@st.composite
def my_thick_layer_strategy(draw):
    return LaminateLayer(
        draw(my_thick_lamina_strategy()),
        float(draw(st.decimals(allow_infinity=False, allow_nan=False))),
        draw(my_orientation_strategy()),
    )


class TestLayer(unittest.TestCase):
    @given(my_thin_layer_strategy())
    def test_Q_vector(self, layer: LaminateLayer) -> None:
        expected_value = np.dot(
            layer.orientation.rotation_matrix, layer.lamina.Q_vector
        )
        np.testing.assert_array_equal(expected_value, layer.Q_vector)

    @given(my_thin_layer_strategy())
    @example(MyTestLayer.layer_30())
    @example(MyTestLayer.test_layer_negative_30())
    def test_Q_matrix(self, layer: LaminateLayer) -> None:
        kwargs = dict()

        vector = layer.Q_vector
        expected_value = [
            [vector[0][0], vector[3][0], vector[4][0]],
            [vector[3][0], vector[1][0], vector[5][0]],
            [vector[4][0], vector[5][0], vector[2][0]],
        ]

        if layer == MyTestLayer.layer_30():
            expected_value = (
                np.array(
                    [
                        [12.08, 3.578, 5.978],
                        [3.578, 2.645, 2.194],
                        [5.978, 2.194, 4.077],
                    ]
                )
                * 1e6
            )
            kwargs["decimal"] = -3

        if layer == MyTestLayer.test_layer_negative_30():
            expected_value = (
                np.array(
                    [
                        [12.08, 3.578, -5.978],
                        [3.578, 2.645, -2.194],
                        [-5.978, -2.194, 4.077],
                    ]
                )
                * 1e6
            )
            kwargs["decimal"] = -3

        actual_value = layer.Q_matrix

        np.testing.assert_array_almost_equal(
            expected_value, actual_value, **kwargs
        )

    @given(my_thick_layer_strategy())
    def test_C_bar(self, layer: LaminateLayer):
        if layer.C_bar is None:
            return None
        expected_value = np.linalg.multi_dot(
            [
                np.linalg.inv(layer.orientation.stress_rotation_matrix_3D),
                layer.lamina.compliance_matrix,
                layer.orientation.strain_rotation_matrix_3D,
            ]
        )
        actual_value = layer.C_bar
        np.testing.assert_array_equal(expected_value, actual_value)

    @given(my_thick_layer_strategy())
    def test_C_entries(self, layer: LaminateLayer):
        if layer.C_bar is None:
            return None
        for i in range(6):
            for j in range(6):
                expected_value = layer.C_bar[i, j]
                actual_value = getattr(layer, f"C{i+1}{j+1}", None)
                if actual_value is not None:
                    self.assertEqual(expected_value, actual_value)

    @given(my_thin_layer_strategy(), my_strain_strategy())
    def test_compute_local_strain(
        self, layer: LaminateLayer, strain: np.ndarray
    ):
        expected_value = layer.orientation.compute_local_strains(strain)
        actual_value = layer.compute_local_strain(strain)
        np.testing.assert_array_equal(expected_value, actual_value)

    @given(my_thin_layer_strategy(), my_strain_strategy())
    def test_compute_local_stress(
        self, layer: LaminateLayer, strain: np.ndarray
    ):
        local_strain = layer.compute_local_strain(strain)
        temperature_delta = 0
        expected_value = layer.lamina.compute_local_stress(
            local_strain, temperature_delta
        )
        actual_value = layer.compute_local_stress(
            local_strain, temperature_delta
        )
        np.testing.assert_array_equal(expected_value, actual_value)

    def test_thermal_expansion_vector(self):
        layer = MyTestLayer.layer_30()
        layer.thickness = 0.02
        expected_value = np.array([[3.773], [11.66], [-13.66]]) * 1e-6
        actual_value = layer.thin_thermal_expansion_vector
        np.testing.assert_array_almost_equal(expected_value, actual_value)

    def test_thick_thermal_expansion_vector(self):
        layer = MyTestLayer.layer_30()
        layer.thickness = 0.02
        expected_value = np.array(
            [
                [3.773e-6],
                [11.66e-6],
                [layer.lamina.alpha2],
                [0],
                [0],
                [-13.66e-6],
            ]
        )
        actual_value = layer.thick_thermal_expansion_vector
        np.testing.assert_array_almost_equal(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()


# End
