import unittest
from test.test_layer import my_thin_layer_strategy
from test.test_loading import (
    MyOnlineExtensionExerciseTestLoad,
    my_load_strategy,
)
from test.test_material import MyOnlineExerciseTestMaterial

import hypothesis.strategies as st
import numpy as np
from hypothesis import example, given

from classical_laminate_theory.laminate import Laminate
from analytical_vessel_analysis.clt.clt.laminate_layer import LaminateLayer
from classical_laminate_theory.loading import LaminateLoad
from classical_laminate_theory.material import Lamina
from classical_laminate_theory.orientation import Orientation


class MyOnlineBendingExerciseTestLoad(LaminateLoad):
    expected_strain = np.array([0, 0, 0, 0.01101, -0.002765, 0]).transpose()

    @classmethod
    def instantiate(cls):
        return cls(Mx=500)


class MyOnlineAsymmetricExerciseTestLoad(LaminateLoad):
    expected_strain = np.array(
        [0.000288, -7.4e-5, 0, 0.006914, -0.00174, 0]
    ).transpose()

    @classmethod
    def instantiate(cls):
        return cls(Mx=500)


class MyOnlineAsymmetricExerciseThermalLoad(LaminateLoad):
    expected_strain = np.vstack(
        np.array([-0.00071, -0.00071, 0, -0.123, 0.123, 0])
    )

    @classmethod
    def instantiate(cls):
        return cls(temperature_delta=-200)


class MyTestMaterial(Lamina):
    @classmethod
    def instantiate(cls):
        return cls(150e3, 20e3, 0.3, 5e3)  # , 0.15


class MyOnlineExerciseFoamMaterial(Lamina):
    # The material is based on the online exercise
    # https://www.youtube.com/watch?v=I0DQH5QCFag

    @classmethod
    def instantiate(cls):
        modulus = 10e3
        v12 = 0.45
        shear_modulus = modulus / (2 * (1 + v12))

        return cls(E1=modulus, E2=modulus, v12=v12, G12=shear_modulus)


class MyAsymmetricTestLaminate(Laminate):
    expected_A = [[112482, 17781.4, 0], [17781.4, 73008.1, 0], [0, 0, 16493.9]]
    expected_B = [[0, 0, 740.132], [0, 0, 740.132], [740.132, 740.132, 0]]
    expected_D = [[16010.6, 953.198, 0], [953.198, 8609.3, 0], [0, 0, 798.704]]

    @classmethod
    def instantiate(cls):
        thickness = 0.15

        return cls(
            [
                LaminateLayer(
                    MyTestMaterial.instantiate(),
                    thickness,
                    Orientation(orientation),
                )
                for orientation in [0, 90, 0, -45, 45, 0, 90, 0]
            ]
        )


class MySymmetricalLaminateTest(Laminate):
    expected_Ex = 85603.7
    expected_Ey = 85603.7
    expected_Gxy = 600.0
    expected_v12 = 0.0706

    @classmethod
    def instantiate(cls):
        thickness = 0.15

        return cls(
            [
                LaminateLayer(
                    MyTestMaterial.instantiate(),
                    thickness,
                    Orientation(orientation),
                )
                for orientation in [0, 90, 0, 90, 90, 0, 90, 0]
            ]
        )


class MyOnlineExtensionExerciseTestLaminate(Laminate):
    # See the online exercise for reference
    # https://www.youtube.com/watch?v=SfhINFjHTjo

    expected_A = (
        np.array([[241.6, 71.56, 0], [71.56, 52.90, 0], [0, 0, 81.54]]) * 1e3
    )
    expected_B = (
        np.array([[0, 0, 740.132], [0, 0, 740.132], [740.132, 740.132, 0]])
        * 1e3
    )
    expected_D = (
        np.array(
            [[16010.6, 953.198, 0], [953.198, 8609.3, 0], [0, 0, 798.704]]
        )
        * 1e3
    )
    expected_Ex = 7.24e6
    expected_Ey = 1.59e6
    expected_Gxy = 4.077e6
    expected_v12 = 1.353
    expected_v21 = 0.296

    @classmethod
    def instantiate(cls):
        thickness = 0.005

        return cls(
            [
                LaminateLayer(
                    MyOnlineExerciseTestMaterial.instantiate(),
                    thickness,
                    Orientation(orientation),
                )
                for orientation in [30, -30, -30, 30]
            ]
        )


class MyOnlineBendingExerciseTestLaminate(Laminate):
    # See the online exercise for reference
    # https://www.youtube.com/watch?v=I0DQH5QCFag

    expected_D = np.array(
        [[45603, 691.2, 0], [691.2, 2751.0, 0], [0, 0, 1821]]
    )

    @classmethod
    def instantiate(cls):
        epoxy_thickness = 0.05
        foam_thickness = 0.25

        carbon_epoxy = LaminateLayer(
            MyOnlineExerciseTestMaterial.instantiate(),
            epoxy_thickness,
            Orientation(0),
        )
        foam = LaminateLayer(
            MyOnlineExerciseFoamMaterial.instantiate(),
            foam_thickness,
            Orientation(0),
        )

        return cls([carbon_epoxy, foam, carbon_epoxy])


class MyOnlineAsymmetricExerciseTestLaminate(Laminate):
    # See the online exercise for reference
    # https://www.youtube.com/watch?v=72IjyZ38_qY

    expected_A = np.array(
        [[3.014e6, 4.658e4, 0], [4.658e4, 1.838e5, 0], [0, 0, 1.209e5]]
    )
    expected_B = np.array(
        [[-1.254e5, -1.847e3, 0], [-1.847e3, -7.450e3, 0], [0, 0, -4.978e3]]
    )
    expected_D = np.array(
        [[7.781e4, 1.175e3, 0], [1.175e3, 4.686e3, 0], [0, 0, 3.105e3]]
    )

    @classmethod
    def instantiate(cls):
        material1 = MyOnlineExerciseTestMaterial.instantiate()
        thickness1 = 0.1
        material2 = MyOnlineExerciseTestMaterial.instantiate()
        thickness2 = 0.05
        foam = MyOnlineExerciseFoamMaterial.instantiate()
        foam_thickness = 0.25
        orientation = Orientation(0)
        return cls(
            [
                LaminateLayer(material1, thickness1, orientation),
                LaminateLayer(foam, foam_thickness, orientation),
                LaminateLayer(material2, thickness2, orientation),
            ]
        )


class MyOnlineExampleThermalLoadedLaminate(Laminate):
    @classmethod
    def instantiate(cls):
        thickness = 0.01

        return cls(
            [
                LaminateLayer(
                    MyOnlineExerciseTestMaterial.instantiate(),
                    thickness,
                    Orientation(orientation),
                )
                for orientation in [0, 90]
            ]
        )


@st.composite
def my_laminate_strategy(draw):
    layers = draw(st.lists(my_thin_layer_strategy(), min_size=1))

    return Laminate(layers)


class TestLaminate(unittest.TestCase):
    @given(my_laminate_strategy())
    def test_layer_thicknesses(self, laminate: Laminate) -> None:
        thicknesses = [layer.thickness for layer in laminate.layers]
        np.testing.assert_array_equal(thicknesses, laminate.layer_thicknesses)

    @given(my_laminate_strategy())
    def test_thickness(self, laminate: Laminate) -> None:
        thickness = 0
        for layer in laminate.layers:
            thickness += layer.thickness
        self.assertEqual(thickness, laminate.thickness)

    @given(my_laminate_strategy())
    def test_z_locations(self, laminate: Laminate) -> None:
        expected_value = [-laminate.thickness / 2]
        for layer in laminate.layers:
            expected_value.append(expected_value[-1] + layer.thickness)
        np.testing.assert_array_equal(expected_value, laminate.z_locations)

    @given(my_laminate_strategy())
    @example(MyAsymmetricTestLaminate.instantiate())
    @example(MyOnlineExtensionExerciseTestLaminate.instantiate())
    @example(MyOnlineAsymmetricExerciseTestLaminate.instantiate())
    def test_A(self, laminate: Laminate) -> None:
        kwargs = dict()

        expected_value = np.zeros((3, 3))
        for k, layer in enumerate(laminate.layers, 1):
            expected_value += layer.Q_matrix * (
                laminate.z_locations[k] - laminate.z_locations[k - 1]
            )

        if laminate == MyAsymmetricTestLaminate.instantiate():
            expected_value = MyAsymmetricTestLaminate.expected_A
            kwargs["decimal"] = -3
        if laminate == MyOnlineExtensionExerciseTestLaminate.instantiate():
            expected_value = MyOnlineExtensionExerciseTestLaminate.expected_A
            kwargs["decimal"] = -3
        if laminate == MyOnlineAsymmetricExerciseTestLaminate.instantiate():
            expected_value = MyOnlineAsymmetricExerciseTestLaminate.expected_A
            kwargs["decimal"] = -3

        actual_value = laminate.A

        np.testing.assert_array_almost_equal(
            expected_value, actual_value, **kwargs
        )

    @given(my_laminate_strategy())
    @example(MyAsymmetricTestLaminate.instantiate())
    @example(MyOnlineAsymmetricExerciseTestLaminate.instantiate())
    def test_B(self, laminate: Laminate) -> None:
        kwargs = dict()

        expected_value = np.zeros((3, 3))
        for k, layer in enumerate(laminate.layers, 1):
            expected_value += (
                0.5
                * layer.Q_matrix
                * (
                    laminate.z_locations[k] ** 2
                    - laminate.z_locations[k - 1] ** 2
                )
            )

        if laminate == MyAsymmetricTestLaminate.instantiate():
            expected_value = MyAsymmetricTestLaminate.expected_B
            kwargs["decimal"] = 0
        if laminate == MyOnlineAsymmetricExerciseTestLaminate.instantiate():
            expected_value = MyOnlineAsymmetricExerciseTestLaminate.expected_B
            kwargs["decimal"] = -3

        actual_value = laminate.B

        np.testing.assert_array_almost_equal(
            expected_value, actual_value, **kwargs
        )

    @given(my_laminate_strategy())
    @example(MyAsymmetricTestLaminate.instantiate())
    @example(MyOnlineBendingExerciseTestLaminate.instantiate())
    @example(MyOnlineAsymmetricExerciseTestLaminate.instantiate())
    def test_D(self, laminate: Laminate) -> None:
        kwargs = dict()

        expected_value = np.zeros((3, 3))
        for k, layer in enumerate(laminate.layers, 1):
            expected_value += (
                1
                / 3
                * layer.Q_matrix
                * (
                    laminate.z_locations[k] ** 3
                    - laminate.z_locations[k - 1] ** 3
                )
            )

        if laminate == MyAsymmetricTestLaminate.instantiate():
            expected_value = MyAsymmetricTestLaminate.expected_D
            kwargs["decimal"] = 0
        if laminate == MyOnlineBendingExerciseTestLaminate.instantiate():
            expected_value = MyOnlineBendingExerciseTestLaminate.expected_D
            kwargs["decimal"] = 0
        if laminate == MyOnlineAsymmetricExerciseTestLaminate.instantiate():
            expected_value = MyOnlineAsymmetricExerciseTestLaminate.expected_D
            kwargs["decimal"] = 0

        actual_value = laminate.D

        np.testing.assert_array_almost_equal(
            expected_value, actual_value, **kwargs
        )

    def test_Ex(self) -> None:
        # Required for the examples in the almost equal function
        kwargs = dict()

        laminates: list[Laminate] = [
            MyOnlineExtensionExerciseTestLaminate.instantiate(),
            MySymmetricalLaminateTest.instantiate(),
        ]
        for laminate in laminates:
            # Catch specific example cases
            if laminate == MySymmetricalLaminateTest.instantiate():
                expected_value = MySymmetricalLaminateTest.expected_Ex
                kwargs["places"] = 1
            if laminate == MyOnlineExtensionExerciseTestLaminate.instantiate():
                expected_value = (
                    MyOnlineExtensionExerciseTestLaminate.expected_Ex
                )
                kwargs["places"] = -4

            actual_value = laminate.Ex

            self.assertAlmostEqual(expected_value, actual_value, **kwargs)

    def test_Ey(self) -> None:
        # Required for the examples in the almost equal function
        kwargs = dict()

        laminates: list[Laminate] = [
            MyOnlineExtensionExerciseTestLaminate.instantiate(),
            MySymmetricalLaminateTest.instantiate(),
        ]
        for laminate in laminates:
            # Catch specific example cases
            if laminate == MySymmetricalLaminateTest.instantiate():
                expected_value = MySymmetricalLaminateTest.expected_Ey
                kwargs["places"] = 1
            if laminate == MyOnlineExtensionExerciseTestLaminate.instantiate():
                expected_value = (
                    MyOnlineExtensionExerciseTestLaminate.expected_Ey
                )
                kwargs["places"] = -4

            actual_value = laminate.Ey

            self.assertAlmostEqual(expected_value, actual_value, **kwargs)

    def test_Gxy(self) -> None:
        # Required for the examples in the almost equal function
        kwargs = dict()

        laminates: list[Laminate] = [
            MyOnlineExtensionExerciseTestLaminate.instantiate(),
        ]
        for laminate in laminates:
            if laminate == MyOnlineExtensionExerciseTestLaminate.instantiate():
                expected_value = (
                    MyOnlineExtensionExerciseTestLaminate.expected_Gxy
                )
                kwargs["places"] = -3

            actual_value = laminate.Gxy

            self.assertAlmostEqual(expected_value, actual_value, **kwargs)

    def test_v_xy(self) -> None:
        # Required for the examples in the almost equal function
        kwargs = dict()

        laminates: list[Laminate] = [
            MyOnlineExtensionExerciseTestLaminate.instantiate(),
            MySymmetricalLaminateTest.instantiate(),
        ]
        for laminate in laminates:
            # Catch specific example cases
            if laminate == MySymmetricalLaminateTest.instantiate():
                expected_value = MySymmetricalLaminateTest.expected_v12
                kwargs["places"] = 3
            if laminate == MyOnlineExtensionExerciseTestLaminate.instantiate():
                expected_value = (
                    MyOnlineExtensionExerciseTestLaminate.expected_v12
                )
                kwargs["places"] = 3

            actual_value = laminate.v_xy

            self.assertAlmostEqual(expected_value, actual_value, **kwargs)

    def test_v_yx(self) -> None:
        # Required for the examples in the almost equal function
        kwargs = dict()

        laminates: list[Laminate] = [
            MyOnlineExtensionExerciseTestLaminate.instantiate(),
        ]
        for laminate in laminates:
            # Catch specific example cases
            if laminate == MyOnlineExtensionExerciseTestLaminate.instantiate():
                expected_value = (
                    MyOnlineExtensionExerciseTestLaminate.expected_v21
                )
                kwargs["places"] = 3

            actual_value = laminate.v_yx

            self.assertAlmostEqual(expected_value, actual_value, **kwargs)

    def test_z_mid_locations(self):
        laminate = MyOnlineExtensionExerciseTestLaminate.instantiate()

        layer = laminate.layers[0]
        lower_z = (
            -0.5 * len(laminate.layers) * layer.thickness
            + 0.5 * layer.thickness
        )
        expected_values = np.array(
            [
                round(lower_z + i * layer.thickness, 4)
                for i in range(len(laminate.layers))
            ]
        )
        actual_value = laminate.z_mid_locations
        np.testing.assert_array_equal(expected_values, actual_value)

    @given(my_laminate_strategy(), my_load_strategy())
    @example(
        MyOnlineExtensionExerciseTestLaminate.instantiate(),
        MyOnlineExtensionExerciseTestLoad.instantiate(),
    )
    def test_compute_laminate_global_strain(
        self, laminate: Laminate, load: LaminateLoad
    ) -> None:
        decimal = {"decimal": 6}
        try:
            actual_value = laminate.compute_global_strain(load)
        except np.linalg.LinAlgError:
            return None

        total_load_vector = laminate.total_load_vector(load)
        expected_value = np.dot(laminate.compliance_matrix, total_load_vector)

        if laminate == MyOnlineExtensionExerciseTestLaminate.instantiate():
            expected_value = MyOnlineExtensionExerciseTestLoad.expected_strain

        np.testing.assert_array_almost_equal(
            expected_value, actual_value, **decimal
        )

    @given(my_laminate_strategy(), my_load_strategy())
    @example(
        MyOnlineExtensionExerciseTestLaminate.instantiate(),
        MyOnlineExtensionExerciseTestLoad.instantiate(),
    )
    def test_compute_total_strain(
        self, laminate: Laminate, load: LaminateLoad
    ):
        try:
            global_strains = laminate.compute_global_strain(load)
        except np.linalg.LinAlgError:
            return None
        actual_values = laminate.compute_total_strains(global_strains)
        expected_values = (
            global_strains[:3] + global_strains[3:] * laminate.z_mid_locations
        )
        for expected, actual in zip(expected_values, actual_values):
            np.testing.assert_array_equal(expected, actual)

    def test_thermal_expansion_vector(self):
        # Create test laminate
        lamina = MyOnlineExerciseTestMaterial.instantiate()
        thickness = 0.01
        layers = [
            LaminateLayer(lamina, thickness, Orientation(rotation))
            for rotation in [30, -30, 0, 0, -30, 30]
        ]
        laminate = Laminate(layers)

        expected_value = np.array([[-1.18], [8.66], [0]]) * 1e-6
        actual_value = laminate.thermal_extension_vector
        np.testing.assert_array_almost_equal(
            expected_value, actual_value, decimal=2
        )

    def test_thermal_moment_vector(self):
        return
        # Create test laminate asymmetric
        lamina = MyOnlineExerciseTestMaterial.instantiate()
        thickness = 0.01
        layers = [
            LaminateLayer(lamina, thickness, Orientation(rotation))
            for rotation in [90, 0]
        ]
        laminate = Laminate(layers)

        expected_value = np.array([[-17.45], [-17.45], [0]]) / (-200)
        actual_value = laminate.thermal_moment_vector
        np.testing.assert_array_almost_equal(
            expected_value, actual_value, decimal=4
        )


if __name__ == "__main__":
    unittest.main()


# End
