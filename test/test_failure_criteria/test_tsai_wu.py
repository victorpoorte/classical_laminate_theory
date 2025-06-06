import math
import unittest
from test.test_failure_criteria.test_failure_criterion import (
    my_strain_vector_strategy,
    stress_state_strategy,
)
from test.test_layer import my_thin_layer_strategy
from test.test_material import lamina_failure_stresses_strategy
from test.test_strategies import my_positive_number_strategy

import numpy as np
from hypothesis import example, given

from classical_laminate_theory.clt.failure_criteria.failure_criterion_protocol import (
    FAILURE_THRESHOLD,
    ROUNDING_DIGITS,
    Failure,
    StressState,
)
from classical_laminate_theory.failure_criteria.tsai_wu import FailureCriterion
from analytical_vessel_analysis.clt.clt.laminate_layer import LaminateLayer
from classical_laminate_theory.material import FailureStresses, LaminaFailureStresses


class MyOnlineExampleStressState(StressState):
    @classmethod
    def instantiate(cls):
        return cls(np.array([[11101], [-1101], [-2251]]))


class MyOnlineExampleLongitudinalFailureStressesTestCase(FailureStresses):
    @classmethod
    def instantiate(cls):
        return cls(*np.array([[-150e3], [300e3], [14e3]]).flatten().tolist())


class MyOnlineExampleTraverseFailureStressesTestCase(FailureStresses):
    @classmethod
    def instantiate(cls):
        return cls(*np.array([[-25e3], [7e3], [14e3]]).flatten().tolist())


class MyOnlineExampleLaminaFailureStressesTestCase(LaminaFailureStresses):
    # These are the expected value, and should be named equal to the
    # method they are testing. First is the expected value, second value
    # is the places accuracy
    compute_F1 = (-3.333e-6, 3)
    compute_F2 = (102.9e-6, 3)
    compute_F11 = (22.22e-12, 3)
    compute_F22 = (5.714e-9, 3)
    compute_F66 = (5.102e-9, 3)
    compute_A = (0.03579, 3)
    compute_B = (-0.1503, 3)
    factor_of_safety = (7.79, 2)
    longitudinal_index = (-0.12207, 3)
    traverse_index = (1.1239491, 3)

    @classmethod
    def instantiate(cls):
        return cls(
            MyOnlineExampleLongitudinalFailureStressesTestCase.instantiate(),
            MyOnlineExampleTraverseFailureStressesTestCase.instantiate(),
        )


class TestTsaiWuFailureCriterion(unittest.TestCase):
    def setUp(self) -> None:
        self.criterion = FailureCriterion()

    def my_almost_equal(
        self, expected_value: float, actual_value: float, places
    ):
        def order_of_magnitude(number):
            if number == 0:
                return 0
            sign = 1 if number > 0 else -1
            return sign * math.floor(math.log(abs(number), 10))

        order = order_of_magnitude(expected_value)
        self.assertAlmostEqual(
            expected_value * 10**-order,
            actual_value * 10**-order,
            places=places,
        )

    def assertAlmostEqualWithExample(
        self, expected_value, actual_value, places
    ):
        self.my_almost_equal(expected_value, actual_value, places["places"])

    def assertComputeMethod(
        self, method_name, stress_state, failure_stresses, expected_value
    ):
        # places = {"places": ROUNDING_DIGITS - 1}
        places = {"places": None}
        # if "index" in method_name or "factor" in method_name:
        #     places["places"] = 1

        # Get or compute the expected value
        if (
            failure_stresses
            == MyOnlineExampleLaminaFailureStressesTestCase.instantiate()
        ):
            actual_value, places["places"] = getattr(
                failure_stresses, method_name
            )
        else:
            compute_method = getattr(self.criterion, method_name)
            # Logic to handle the different amount of arguments required
            # for the methods that are to be tested. This has been done
            # to reduce the amount of code repetition.
            if stress_state is None:
                actual_value = compute_method(failure_stresses)
            else:
                actual_value = compute_method(stress_state, failure_stresses)

        self.assertAlmostEqualWithExample(expected_value, actual_value, places)

    @given(lamina_failure_stresses_strategy())
    @example(MyOnlineExampleLaminaFailureStressesTestCase.instantiate())
    def test_compute_F11(
        self, failure_stresses: LaminaFailureStresses
    ) -> None:
        # Compute the expected value
        expected_value = -1 / (
            failure_stresses.longitudinal.tension
            * failure_stresses.longitudinal.compression
        )

        self.assertComputeMethod(
            "compute_F11", None, failure_stresses, expected_value
        )

    @given(lamina_failure_stresses_strategy())
    @example(MyOnlineExampleLaminaFailureStressesTestCase.instantiate())
    def test_compute_F22(
        self, failure_stresses: LaminaFailureStresses
    ) -> None:
        # Compute the expected value
        expected_value = -1 / (
            failure_stresses.traverse.tension
            * failure_stresses.traverse.compression
        )

        self.assertComputeMethod(
            "compute_F22", None, failure_stresses, expected_value
        )

    @given(lamina_failure_stresses_strategy())
    @example(MyOnlineExampleLaminaFailureStressesTestCase.instantiate())
    def test_compute_F66(
        self, failure_stresses: LaminaFailureStresses
    ) -> None:
        expected_value = 1 / failure_stresses.longitudinal.shear**2

        self.assertComputeMethod(
            "compute_F66", None, failure_stresses, expected_value
        )

    @given(stress_state_strategy(), lamina_failure_stresses_strategy())
    @example(
        MyOnlineExampleStressState.instantiate(),
        MyOnlineExampleLaminaFailureStressesTestCase.instantiate(),
    )
    def test_compute_A(
        self,
        stress_state: StressState,
        failure_stresses: LaminaFailureStresses,
    ) -> None:
        expected_value = (
            self.criterion._compute_F11(failure_stresses)
            * stress_state.longitudinal**2
            + self.criterion._compute_F22(failure_stresses)
            * stress_state.traverse**2
            + self.criterion._compute_F66(failure_stresses)
            * stress_state.shear**2
            - self.criterion._compute_F11(failure_stresses)
            * stress_state.longitudinal
            * stress_state.traverse
        )

        self.assertComputeMethod(
            "compute_A", stress_state, failure_stresses, expected_value
        )

    @given(stress_state_strategy(), lamina_failure_stresses_strategy())
    @example(
        MyOnlineExampleStressState.instantiate(),
        MyOnlineExampleLaminaFailureStressesTestCase.instantiate(),
    )
    def test_compute_B(
        self,
        stress_state: StressState,
        failure_stresses: LaminaFailureStresses,
    ) -> None:
        # Compute the expected value
        expected_value = (
            self.criterion._compute_F1(failure_stresses)
            * stress_state.longitudinal
            + self.criterion._compute_F2(failure_stresses)
            * stress_state.traverse
        )

        self.assertComputeMethod(
            "compute_B", stress_state, failure_stresses, expected_value
        )

    @given(my_strain_vector_strategy(), my_thin_layer_strategy())
    def test_factor_of_safety(
        self, global_strain: np.ndarray, layer: LaminateLayer
    ) -> None:
        temperature_delta = 0
        stress_state = self.criterion._compute_layer_stress_state(
            layer, global_strain, temperature_delta
        )

        a = self.criterion._compute_A(
            stress_state, layer.lamina.failure_stresses
        )
        b = self.criterion._compute_B(
            stress_state, layer.lamina.failure_stresses
        )
        # It might occur that the random values yield impossible solutions
        if a == 0 or (b**2 + 4 * a) < 0:
            return None
        expected_value = round(
            (math.sqrt(b**2 + 4 * a) - b) / (2 * a), ROUNDING_DIGITS
        )
        actual_value = self.criterion.factor_of_safety(
            global_strain, layer, temperature_delta
        )
        self.assertEqual(expected_value, actual_value)

    @given(stress_state_strategy(), lamina_failure_stresses_strategy())
    @example(
        StressState(np.array([[86474], [-8574], [17538]])),
        MyOnlineExampleLaminaFailureStressesTestCase.instantiate(),
    )
    def test_longitudinal_index(
        self,
        stress_state: StressState,
        failure_stresses: LaminaFailureStresses,
    ) -> None:
        expected_value = round(
            self.criterion._compute_F1(failure_stresses)
            * stress_state.longitudinal
            + self.criterion._compute_F11(failure_stresses)
            * stress_state.longitudinal**2,
            ROUNDING_DIGITS,
        )

        self.assertComputeMethod(
            "longitudinal_index",
            stress_state,
            failure_stresses,
            expected_value,
        )

    @given(stress_state_strategy(), lamina_failure_stresses_strategy())
    @example(
        StressState(np.array([[86474], [-8574], [17538]])),
        MyOnlineExampleLaminaFailureStressesTestCase.instantiate(),
    )
    def test_traverse_index(
        self,
        stress_state: StressState,
        failure_stresses: LaminaFailureStresses,
    ) -> None:
        expected_value = round(
            self.criterion._compute_F2(failure_stresses) * stress_state.traverse
            + self.criterion._compute_F22(failure_stresses)
            * stress_state.traverse**2
            + self.criterion._compute_F66(failure_stresses)
            * stress_state.shear**2
            - self.criterion._compute_F11(failure_stresses)
            * stress_state.longitudinal
            * stress_state.traverse,
            ROUNDING_DIGITS,
        )

        self.assertComputeMethod(
            "traverse_index", stress_state, failure_stresses, expected_value
        )

    @given(stress_state_strategy(), lamina_failure_stresses_strategy())
    def test_compute_F12(
        self, stress_state: StressState, failure_stresses: FailureStresses
    ):
        expected_value = -0.5 * self.criterion._compute_F11(failure_stresses)
        actual_value = self.criterion._compute_F12(failure_stresses)
        self.assertEqual(expected_value, actual_value)

    @given(
        my_strain_vector_strategy(),
        my_thin_layer_strategy(),
        my_positive_number_strategy(),
    )
    def test_failure_indices(
        self,
        global_strain: np.ndarray,
        layer: LaminateLayer,
        temperature_delta: float,
    ):
        stress_state = self.criterion._compute_layer_stress_state(
            layer, global_strain, temperature_delta
        )
        long_index = self.criterion._longitudinal_index(
            stress_state, layer.lamina.failure_stresses
        )
        trav_index = self.criterion._traverse_index(
            stress_state, layer.lamina.failure_stresses
        )
        failure = long_index + trav_index >= FAILURE_THRESHOLD
        long_failure = failure and long_index >= trav_index
        trav_failure = failure and not long_failure
        expected_value = Failure(long_failure, trav_failure)
        actual_value = self.criterion.failure(
            global_strain, layer, temperature_delta
        )
        self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()


# End
