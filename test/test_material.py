import unittest
from classical_laminate_theory.laminate import Laminate
from classical_laminate_theory.orientation import Orientation
from analytical_vessel_analysis.clt.clt.laminate_layer import LaminateLayer
from test.test_strategies import (
    my_list_of_numbers_strategy,
    my_positive_number_strategy,
)

import hypothesis.strategies as st
import numpy as np
from hypothesis import example, given

from classical_laminate_theory.material import (
    FailureStresses,
    Lamina,
    LaminaFailureStresses,
)


@st.composite
def failure_stresses_strategy(draw):
    number_of_vars = 3
    variables = draw(my_list_of_numbers_strategy(number_of_vars))
    variables[0] *= -1
    return FailureStresses(*variables)


@st.composite
def lamina_failure_stresses_strategy(draw):
    failure_loads = draw(
        st.lists(failure_stresses_strategy(), min_size=2, max_size=2)
    )
    return LaminaFailureStresses(*failure_loads)


class MyOnlineExerciseTestMaterial(Lamina):
    # The material is based on the online exercise
    # https://www.youtube.com/watch?v=SfhINFjHTjo

    @classmethod
    def instantiate(cls):
        return cls(
            E1=20e6,
            E2=1.2e6,
            E3=1.2e6,
            v12=0.25,
            G12=0.8e6,
            G13=3.3e6,
            G23=1.1e6,
            v23=1.14,
            alpha1=-0.17e-6,
            alpha2=15.6e-6,
        )

    @property
    def strain_vector(self):
        return np.array([0.001422, -0.002640, -0.007036])

    @property
    def expected_stress_vector(self):
        return np.array([27752, -2752, -5629])

    @property
    def expected_v21(self):
        return 0.0150


@st.composite
def my_lamina_strategy(draw):
    lamina_variables = 4
    return Lamina(
        *draw(my_list_of_numbers_strategy(lamina_variables)),
        failure_stresses=draw(lamina_failure_stresses_strategy()),
        alpha1=draw(my_positive_number_strategy()),
        alpha2=draw(my_positive_number_strategy())
    )


@st.composite
def my_thick_lamina_strategy(draw):
    lamina_variables = 11
    return Lamina(*draw(my_list_of_numbers_strategy(lamina_variables)))


@st.composite
def my_strain_strategy(draw):
    array_size = 3
    return np.array(draw(my_list_of_numbers_strategy(array_size)))


class TestFailureStresses(unittest.TestCase):
    def test_check_compression_sing(self):
        entries = [
            "Positive compression.",
            "Better check your material definition...",
        ]
        with self.assertRaises(ValueError) as cm:
            FailureStresses(1, 2, 3)
        self.assertEqual(str(cm.exception), " ".join(entries))


class TestLamina(unittest.TestCase):
    @given(my_lamina_strategy())
    @example(MyOnlineExerciseTestMaterial.instantiate())
    def test_v21(self, lamina: Lamina) -> None:
        expected_value = lamina.E2 * lamina.v12 / lamina.E1
        actual_value = lamina.v21

        if lamina == MyOnlineExerciseTestMaterial.instantiate():
            expected_value = (
                MyOnlineExerciseTestMaterial.instantiate().expected_v21
            )
        self.assertEqual(expected_value, actual_value)

    @given(my_lamina_strategy())
    @example(MyOnlineExerciseTestMaterial.instantiate())
    def test_Q_xx(self, lamina: Lamina) -> None:
        expected_value = lamina.E1 / (1 - lamina.v12 * lamina.v21)
        actual_value = lamina.Q_xx
        self.assertEqual(expected_value, actual_value)

        if lamina == MyOnlineExerciseTestMaterial.instantiate():
            expected_value = 20.08e6
            self.assertAlmostEqual(expected_value, actual_value, places=-5)

    @given(my_lamina_strategy())
    @example(MyOnlineExerciseTestMaterial.instantiate())
    def test_Q_yy(self, lamina: Lamina) -> None:
        expected_value = lamina.E2 / (1 - lamina.v12 * lamina.v21)
        actual_value = lamina.Q_yy
        self.assertEqual(expected_value, actual_value)

        if lamina == MyOnlineExerciseTestMaterial.instantiate():
            expected_value = 1.205e6
            self.assertAlmostEqual(expected_value, actual_value, places=-6)

    @given(my_lamina_strategy())
    @example(MyOnlineExerciseTestMaterial.instantiate())
    def test_Q_xy(self, lamina: Lamina) -> None:
        expected_value = lamina.v12 * lamina.E2 / (1 - lamina.v12 * lamina.v21)
        actual_value = lamina.Q_xy
        self.assertAlmostEqual(expected_value, actual_value)

        if lamina == MyOnlineExerciseTestMaterial.instantiate():
            expected_value = 0.3011e6
            self.assertAlmostEqual(expected_value, actual_value, places=-6)

    @given(my_lamina_strategy())
    @example(MyOnlineExerciseTestMaterial.instantiate())
    def test_Q_ss(self, lamina: Lamina) -> None:
        expected_value = lamina.G12
        actual_value = lamina.Q_ss
        self.assertEqual(expected_value, actual_value)

        if lamina == MyOnlineExerciseTestMaterial.instantiate():
            expected_value = 0.8e6
            self.assertAlmostEqual(expected_value, actual_value, places=-6)

    @given(my_lamina_strategy())
    def test_Q_vector(self, lamina: Lamina):
        expected_value = np.array(
            [[lamina.Q_xx], [lamina.Q_yy], [lamina.Q_xy], [lamina.Q_ss]]
        )
        actual_value = lamina.Q_vector
        np.testing.assert_array_equal(expected_value, actual_value)

    @given(my_lamina_strategy(), my_strain_strategy())
    @example(
        MyOnlineExerciseTestMaterial.instantiate(),
        MyOnlineExerciseTestMaterial.instantiate().strain_vector,
    )
    def test_compute_local_stress(
        self, lamina: Lamina, strain_vector: np.ndarray
    ):
        expected_value = np.dot(lamina.Q_matrix, strain_vector)

        if lamina == MyOnlineExerciseTestMaterial.instantiate():
            expected_value = (
                MyOnlineExerciseTestMaterial.instantiate().expected_stress_vector
            )

        temperature_delta = 0
        actual_value = lamina.compute_local_stress(
            strain_vector, temperature_delta
        )

        # print(expected_stresses)
        np.testing.assert_array_almost_equal(
            expected_value, actual_value, decimal=0
        )

    # @given(my_thick_lamina_strategy())
    # def test_v13(self, lamina: Lamina) -> None:
    #     expected_value = lamina.v12
    #     actual_value = lamina.v13
    #     self.assertEqual(expected_value, actual_value)

    @given(my_thick_lamina_strategy())
    def test_S11(self, lamina: Lamina) -> None:
        expected_value = 1 / lamina.E1
        actual_value = lamina.S11
        self.assertEqual(expected_value, actual_value)

    @given(my_thick_lamina_strategy())
    def test_S22(self, lamina: Lamina) -> None:
        expected_value = 1 / lamina.E2
        actual_value = lamina.S22
        self.assertEqual(expected_value, actual_value)

    @given(my_thick_lamina_strategy())
    def test_S33(self, lamina: Lamina) -> None:
        expected_value = 1 / lamina.E3
        actual_value = lamina.S33
        self.assertEqual(expected_value, actual_value)

    @given(my_thick_lamina_strategy())
    def test_S44(self, lamina: Lamina) -> None:
        expected_value = 1 / lamina.G23
        actual_value = lamina.S44
        self.assertEqual(expected_value, actual_value)

    @given(my_thick_lamina_strategy())
    def test_S55(self, lamina: Lamina) -> None:
        expected_value = 1 / lamina.G13
        actual_value = lamina.S55
        self.assertEqual(expected_value, actual_value)

    @given(my_thick_lamina_strategy())
    def test_S66(self, lamina: Lamina) -> None:
        expected_value = 1 / lamina.G12
        actual_value = lamina.S66
        self.assertEqual(expected_value, actual_value)

    @given(my_thick_lamina_strategy())
    def test_S(self, lamina: Lamina) -> None:
        expected_value = np.array(
            [
                [lamina.S11, lamina.S12, lamina.S13, 0, 0, 0],
                [lamina.S12, lamina.S22, lamina.S23, 0, 0, 0],
                [lamina.S13, lamina.S23, lamina.S33, 0, 0, 0],
                [0, 0, 0, lamina.S44, 0, 0],
                [0, 0, 0, 0, lamina.S55, 0],
                [0, 0, 0, 0, 0, lamina.S66],
            ]
        )
        actual_value = lamina.stiffness_matrix
        np.testing.assert_array_equal(expected_value, actual_value)

    @given(my_thick_lamina_strategy())
    def test_C(self, lamina: Lamina) -> None:
        stiffness_matrix = np.array(
            [
                [lamina.S11, lamina.S12, lamina.S13, 0, 0, 0],
                [lamina.S12, lamina.S22, lamina.S23, 0, 0, 0],
                [lamina.S13, lamina.S23, lamina.S33, 0, 0, 0],
                [0, 0, 0, lamina.S44, 0, 0],
                [0, 0, 0, 0, lamina.S55, 0],
                [0, 0, 0, 0, 0, lamina.S66],
            ]
        )
        try:
            expected_value = np.linalg.inv(stiffness_matrix)
        except np.linalg.LinAlgError:
            return None
        actual_value = lamina.compliance_matrix
        np.testing.assert_array_equal(expected_value, actual_value)

    def test_thermal_expansion_vector(self):
        lamina = MyOnlineExerciseTestMaterial.instantiate()
        expected_value = np.array([[lamina.alpha1], [lamina.alpha2], [0]])
        actual_value = lamina.thermal_expansion_vector
        np.testing.assert_array_equal(expected_value, actual_value)

    @given(my_thick_lamina_strategy())
    def test_get_compliance_matrix(self, lamina: Lamina):
        if lamina.compliance_matrix is not None:
            for strain_size in [3, 6]:
                strain_vector = np.ones((strain_size,))
                actual_value = lamina._get_compliance_matrix(
                    strain_vector
                ).shape
                expected_value = (strain_size, strain_size)
                self.assertEqual(expected_value, actual_value)

        strain_size = 7
        strain_vector = np.ones((strain_size,))
        with self.assertRaises(ValueError) as ctx:
            lamina._get_compliance_matrix(strain_vector)
        self.assertEqual(str(ctx.exception), "Invalid strain vector shape...")

    def test_reshape_strain_vector(self):
        strain_vector = np.ones((6,))
        expected_value = strain_vector
        actual_value = Lamina.reshape_strain_vector(strain_vector)
        np.testing.assert_array_equal(expected_value, actual_value)
        strain_vector = np.ones((6,))
        strain_vector[-1] = 3.3
        expected_value[-1] = 3.3
        actual_value = Lamina.reshape_strain_vector(strain_vector)
        np.testing.assert_array_equal(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()


# End
