import copy
import unittest
from test.test_material_degraders.test_material_degrader import (
    my_lamina_strategy,
    my_laminate_failure_stresses_strategy,
)

from hypothesis import example, given

from clt.clt.failure_criteria.failure_criterion_protocol import Failure
from clt.material import Lamina, LaminaFailureStresses
from clt.clt.material_degraders.degrader_protocol import DEGRADATION_FACTOR
from clt.material_degraders.tsai_wu import MaterialDegrader


class TestMaterialDegrader(unittest.TestCase):
    def setUp(self) -> None:
        self.degrader = MaterialDegrader()

    @given(my_laminate_failure_stresses_strategy())
    @example(None)
    def test_longitudinal_failure_properties(
        self, failure_stresses: LaminaFailureStresses
    ):
        self.assertEqual(
            failure_stresses,
            self.degrader.degrade_longitudinal_failure_properties(
                failure_stresses
            ),
        )
        # if failure_stresses is None:
        #     self.assertIsNone(
        #         self.degrader.degrade_longitudinal_failure_properties(failure_stresses)
        #     )
        #     return None

        # longitudinal = copy.copy(failure_stresses.longitudinal)
        # longitudinal.compression /= DEGRADATION_FACTOR
        # longitudinal.tension /= DEGRADATION_FACTOR
        # longitudinal.shear /= DEGRADATION_FACTOR
        # traverse = copy.copy(failure_stresses.traverse)
        # traverse.shear *= DEGRADATION_FACTOR

        # expected_value = LaminaFailureStresses(longitudinal, traverse)
        # actual_value = self.degrader.degrade_longitudinal_failure_properties(
        #     failure_stresses
        # )
        # self.assertEqual(expected_value, actual_value)

    @given(my_laminate_failure_stresses_strategy())
    @example(None)
    def test_traverse_failure_properties(
        self, failure_stresses: LaminaFailureStresses
    ):
        if failure_stresses is None:
            self.assertIsNone(
                self.degrader.degrade_traverse_failure_properties(
                    failure_stresses
                )
            )
            return None

        longitudinal = copy.copy(failure_stresses.longitudinal)
        longitudinal.compression *= (
            self.degrader.compression_degradation_factor
        )
        traverse = copy.copy(failure_stresses.traverse)

        expected_value = LaminaFailureStresses(longitudinal, traverse)
        actual_value = self.degrader.degrade_traverse_failure_properties(
            failure_stresses
        )
        self.assertEqual(expected_value, actual_value)

    @given(my_lamina_strategy())
    def test_degrade_longitudinal_failure(self, lamina: Lamina):
        lamina.degraded_fibre = True
        self.assertEqual(
            lamina, self.degrader.degrade_longitudinal_failure(lamina)
        )

        lamina.degraded_fibre = False
        actual_value = copy.copy(lamina)
        actual_value.E1 *= DEGRADATION_FACTOR
        actual_value.v12 = self.degrader.degraded_poisson
        actual_value.G12 *= DEGRADATION_FACTOR
        actual_value.E3 *= DEGRADATION_FACTOR
        actual_value.G13 *= DEGRADATION_FACTOR
        actual_value.failure_stresses = (
            self.degrader.degrade_longitudinal_failure_properties(
                lamina.failure_stresses
            )
        )
        actual_value.degraded_fibre = True
        expected_value = self.degrader.degrade_longitudinal_failure(lamina)
        self.assertEqual(actual_value, expected_value)

    @given(my_lamina_strategy())
    def test_degrade_traverse_failure(self, lamina: Lamina):
        lamina.degraded_matrix = True
        self.assertEqual(
            lamina, self.degrader.degrade_traverse_failure(lamina)
        )

        lamina.degraded_matrix = False
        actual_value = copy.copy(lamina)
        actual_value.E2 *= DEGRADATION_FACTOR
        actual_value.E3 *= DEGRADATION_FACTOR
        actual_value.v12 = self.degrader.degraded_poisson
        actual_value.G12 *= DEGRADATION_FACTOR
        actual_value.G13 *= DEGRADATION_FACTOR
        actual_value.failure_stresses = (
            self.degrader.degrade_traverse_failure_properties(
                lamina.failure_stresses
            )
        )
        actual_value.degraded_matrix = True
        expected_value = self.degrader.degrade_traverse_failure(lamina)
        self.assertEqual(actual_value, expected_value)

    @given(my_lamina_strategy())
    def test_degrade_material(self, material: Lamina):
        failure = Failure(True, False)
        expected = self.degrader.degrade_longitudinal_failure(material)
        actual = self.degrader.degrade_material(material, failure)
        self.assertEqual(expected, actual)

        failure = Failure(True, True)
        expected = self.degrader.degrade_longitudinal_failure(material)
        actual = self.degrader.degrade_material(material, failure)
        self.assertEqual(expected, actual)

        failure = Failure(False, False)
        expected = material
        actual = self.degrader.degrade_material(material, failure)
        self.assertEqual(expected, actual)

        failure = Failure(False, True)
        expected = self.degrader.degrade_traverse_failure(material)
        actual = self.degrader.degrade_material(material, failure)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()


# End
