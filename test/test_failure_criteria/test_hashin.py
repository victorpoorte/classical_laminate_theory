import math
import unittest

import numpy as np
from hypothesis import given
from test_failure_criteria.test_failure_criterion import (
    my_strain_vector_strategy,
    stress_state_strategy,
)
from test_layer import my_thin_layer_strategy
from test_material import (
    failure_stresses_strategy,
    lamina_failure_stresses_strategy,
)

from clt.clt.failure_criteria.failure_criterion_protocol import StressState
from clt.clt.failure_criteria.hashin import FailureCriterion
from analytical_vessel_analysis.clt.clt.laminate_layer import LaminateLayer
from clt.material import FailureStresses, LaminaFailureStresses


class TestHashinFailureCriterion(unittest.TestCase):
    def setUp(self) -> None:
        self.criterion = FailureCriterion()

    @given(stress_state_strategy(), failure_stresses_strategy())
    def test_longitudinal_tension_index(
        self, stress_state: StressState, failure_stresses: FailureStresses
    ):
        if stress_state.longitudinal >= 0:
            alpha = 1
            term1 = (stress_state.longitudinal / failure_stresses.tension) ** 2
            term2 = (stress_state.shear / failure_stresses.shear) ** 2
            expected_value = term1 + alpha * term2
            self.assertEqual(
                expected_value,
                self.criterion._longitudinal_tension_factor(
                    stress_state, failure_stresses
                ),
            )

    @given(stress_state_strategy(), failure_stresses_strategy())
    def test_longitudinal_compression_index(
        self, stress_state: StressState, failure_loads: FailureStresses
    ):
        if stress_state.longitudinal < 0:
            expected_value = (
                stress_state.longitudinal / failure_loads.compression
            ) ** 2
            self.assertEqual(
                expected_value,
                self.criterion._longitudinal_compression_factor(
                    stress_state, failure_loads
                ),
            )

    @given(stress_state_strategy(), lamina_failure_stresses_strategy())
    def test_traverse_tension_index(
        self,
        stress_state: StressState,
        failure_stresses: LaminaFailureStresses,
    ):
        if stress_state.longitudinal >= 0:
            term1 = (
                stress_state.traverse / failure_stresses.traverse.tension
            ) ** 2
            term2 = (
                stress_state.shear / failure_stresses.longitudinal.shear
            ) ** 2
            expected_value = term1 + term2
            self.assertEqual(
                expected_value,
                self.criterion._traverse_tension_factor(
                    stress_state, failure_stresses
                ),
            )

    @given(stress_state_strategy(), lamina_failure_stresses_strategy())
    def test_traverse_compression_index(
        self,
        stress_state: StressState,
        failure_stresses: LaminaFailureStresses,
    ):
        term1 = (
            stress_state.traverse / (2 * failure_stresses.traverse.shear)
        ) ** 2
        term2 = (
            (
                failure_stresses.traverse.compression
                / (2 * failure_stresses.traverse.shear)
            )
            ** 2
            - 1
        ) * (stress_state.traverse / failure_stresses.traverse.compression)
        term3 = (stress_state.shear / failure_stresses.longitudinal.shear) ** 2
        expected_value = term1 + term2 + term3
        self.assertEqual(
            expected_value,
            self.criterion._traverse_compression_factor(
                stress_state, failure_stresses
            ),
        )

    @given(stress_state_strategy(), lamina_failure_stresses_strategy())
    def test_longitudinal_index(
        self,
        stress_state: StressState,
        failure_stresses: LaminaFailureStresses,
    ):
        if stress_state.longitudinal >= 0:
            method = self.criterion._longitudinal_tension_factor
        else:
            method = self.criterion._longitudinal_compression_factor
        expected_value = method(stress_state, failure_stresses.longitudinal)
        actual_value = self.criterion._longitudinal_factor(
            stress_state, failure_stresses
        )
        self.assertEqual(expected_value, actual_value)

    @given(stress_state_strategy(), lamina_failure_stresses_strategy())
    def test_traverse_index(
        self,
        stress_state: StressState,
        failure_stresses: LaminaFailureStresses,
    ):
        if stress_state.traverse > 0:
            method = self.criterion._traverse_tension_factor
        else:
            method = self.criterion._traverse_compression_factor
        expected_value = method(stress_state, failure_stresses)
        actual_value = self.criterion._traverse_factor(
            stress_state, failure_stresses
        )
        self.assertEqual(expected_value, actual_value)

    @given(stress_state_strategy(), lamina_failure_stresses_strategy())
    def test_longitudinal_tension_factor_of_safety(
        self,
        stress_state: StressState,
        failure_stresses: LaminaFailureStresses,
    ):
        if stress_state.longitudinal < 0:
            return None

        failure_index = self.criterion._longitudinal_tension_factor(
            stress_state, failure_stresses.longitudinal
        )
        if failure_index == 0:
            return None
        expected_value = math.sqrt(1 / failure_index)
        actual_value = self.criterion._longitudinal_tension_factor_of_safety(
            stress_state, failure_stresses.longitudinal
        )
        self.assertEqual(expected_value, actual_value)

    @given(stress_state_strategy(), lamina_failure_stresses_strategy())
    def test_longitudinal_compression_factor_of_safety(
        self,
        stress_state: StressState,
        failure_stresses: LaminaFailureStresses,
    ):
        if stress_state.longitudinal >= 0:
            return None

        failure_index = self.criterion._longitudinal_compression_factor(
            stress_state, failure_stresses.longitudinal
        )
        if failure_index == 0:
            return None
        expected_value = math.sqrt(1 / failure_index)
        actual_value = (
            self.criterion._longitudinal_compression_factor_of_safety(
                stress_state, failure_stresses.longitudinal
            )
        )
        self.assertEqual(expected_value, actual_value)

    @given(stress_state_strategy(), lamina_failure_stresses_strategy())
    def test_traverse_tension_factor_of_safety(
        self,
        stress_state: StressState,
        failure_stresses: LaminaFailureStresses,
    ):
        if stress_state.longitudinal < 0:
            return None

        failure_index = self.criterion._traverse_tension_factor(
            stress_state, failure_stresses
        )
        if failure_index == 0:
            return None
        expected_value = math.sqrt(1 / failure_index)
        actual_value = self.criterion._traverse_tension_factor_of_safety(
            stress_state, failure_stresses
        )
        self.assertEqual(expected_value, actual_value)

    @given(stress_state_strategy(), lamina_failure_stresses_strategy())
    def test_traverse_compression_factor_of_safety(
        self,
        stress_state: StressState,
        failure_stresses: LaminaFailureStresses,
    ):
        if stress_state.longitudinal >= 0:
            return None

        a = self.criterion._determine_a_factor_of_safety(
            stress_state, failure_stresses
        )
        b = self.criterion._determine_b_factor_of_safety(
            stress_state, failure_stresses
        )
        if a == 0:
            return None

        expected_value = (math.sqrt(b**2 + 4 * a) - b) / (2 * a)

        actual_value = self.criterion._traverse_compression_factor_of_safety(
            stress_state, failure_stresses
        )

        self.assertEqual(expected_value, actual_value)

    @given(stress_state_strategy(), lamina_failure_stresses_strategy())
    def test_longitudinal_factor_of_safety(
        self,
        stress_state: StressState,
        failure_stresses: LaminaFailureStresses,
    ):
        if stress_state.longitudinal >= 0:
            method = self.criterion._longitudinal_tension_factor_of_safety
        else:
            method = self.criterion._longitudinal_compression_factor_of_safety
        try:
            expected_value = method(
                stress_state, failure_stresses.longitudinal
            )
        except ZeroDivisionError:
            return None
        actual_value = self.criterion.longitudinal_factor_of_safety(
            stress_state, failure_stresses.longitudinal
        )
        self.assertEqual(expected_value, actual_value)

    @given(stress_state_strategy(), lamina_failure_stresses_strategy())
    def test_traverse_factor_of_safety(
        self,
        stress_state: StressState,
        failure_stresses: LaminaFailureStresses,
    ):
        if stress_state.traverse >= 0:
            method = self.criterion._traverse_tension_factor_of_safety
        else:
            method = self.criterion._traverse_compression_factor_of_safety
        try:
            expected_value = method(stress_state, failure_stresses)
        except ZeroDivisionError:
            return None
        actual_value = self.criterion.traverse_factor_of_safety(
            stress_state, failure_stresses
        )
        self.assertEqual(expected_value, actual_value)

    @given(my_strain_vector_strategy(), my_thin_layer_strategy())
    def test_factor_of_safety(
        self, global_strain: np.ndarray, layer: LaminateLayer
    ):
        temperature_delta = 0
        stress_state = self.criterion._compute_layer_stress_state(
            layer, global_strain, temperature_delta
        )
        try:
            traverse = self.criterion.traverse_factor_of_safety(
                stress_state, layer.lamina.failure_stresses
            )
            longitudinal = self.criterion.longitudinal_factor_of_safety(
                stress_state, layer.lamina.failure_stresses.longitudinal
            )
        except ZeroDivisionError:
            return None

        expected_value = min([traverse, longitudinal])
        actual_value = self.criterion.factor_of_safety(
            global_strain, layer, temperature_delta
        )
        self.assertEqual(expected_value, actual_value)

    def test_longitudinal_failure(self):
        long_index = 1.0
        trav_index = 0.9
        self.assertTrue(
            self.criterion.longitudinal_failure(long_index, trav_index)
        )

        long_index = 1.0
        trav_index = 1.1
        self.assertTrue(
            self.criterion.longitudinal_failure(long_index, trav_index)
        )

        long_index = 0.9
        trav_index = 0.9
        self.assertFalse(
            self.criterion.longitudinal_failure(long_index, trav_index)
        )


if __name__ == "__main__":
    unittest.main()


# End
