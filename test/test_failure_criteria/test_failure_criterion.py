import unittest
from test.test_layer import my_thin_layer_strategy
from test.test_strategies import (
    my_list_of_numbers_strategy,
    my_positive_number_strategy,
)
from unittest.mock import patch

import hypothesis.strategies as st
import numpy as np
from hypothesis import given

from clt.clt.failure_criteria.failure_criterion_protocol import (
    Failure,
    FailureCriterion,
    StressState,
)
from analytical_vessel_analysis.clt.clt.laminate_layer import LaminateLayer


class FailureCriterion(FailureCriterion):
    ...


@st.composite
def stress_state_strategy(draw):
    number_of_vars = 3
    return StressState(
        np.array(
            [
                [value]
                for value in draw(
                    my_list_of_numbers_strategy(
                        number_of_vars, min_value=False
                    )
                )
            ]
        )
    )


@st.composite
def my_strain_vector_strategy(draw):
    strain_entries = 3
    return np.array(
        [var for var in draw(my_list_of_numbers_strategy(strain_entries))]
    )


class TestFailureCriterion(unittest.TestCase):
    @patch.multiple(FailureCriterion, __abstractmethods__=set())
    def setUp(self) -> None:
        self.criterion = FailureCriterion()

    @given(st.decimals(allow_nan=False, allow_infinity=False))
    def test_tension(self, stress: float):
        stress = float(stress)

        if stress >= 0:
            self.assertTrue(self.criterion._tension(stress))
        else:
            self.assertFalse(self.criterion._tension(stress))

    @given(
        my_thin_layer_strategy(),
        my_strain_vector_strategy(),
        my_positive_number_strategy(),
    )
    def test_compute_layer_stress_state(
        self,
        layer: LaminateLayer,
        global_strain: np.ndarray,
        temperature_delta: float,
    ):
        expected_value = StressState(
            layer.compute_local_stress(
                layer.compute_local_strain(global_strain), temperature_delta
            )
        )
        actual_value = self.criterion._compute_layer_stress_state(
            layer, global_strain, temperature_delta
        )
        self.assertEqual(expected_value, actual_value)


class TestFailure(unittest.TestCase):
    def test_failure(self):
        failure = Failure(True, False)
        self.assertTrue(failure.failure)

        failure = Failure(True, True)
        self.assertTrue(failure.failure)

        failure = Failure(False, True)
        self.assertTrue(failure.failure)

        failure = Failure(False, False)
        self.assertFalse(failure.failure)


if __name__ == "__main__":
    unittest.main()


# End
