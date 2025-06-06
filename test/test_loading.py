import unittest
from test.test_strategies import my_list_of_numbers_strategy

import hypothesis.strategies as st
import numpy as np
from hypothesis import given

from classical_laminate_theory.loading import LaminateLoad

LOAD_KEYS = {"Nx", "Ny", "Nxy", "Mx", "My", "Mxy"}


class MyOnlineExtensionExerciseTestLoad(LaminateLoad):
    expected_strain = np.array([[0.003453], [-0.004671], [0], [0], [0], [0]])

    @classmethod
    def instantiate(cls):
        return cls(Nx=500)


@st.composite
def my_load_strategy(draw):
    load_entries = 6
    values = draw(my_list_of_numbers_strategy(load_entries))

    return LaminateLoad(
        **{key: float(value) for key, value in zip(LOAD_KEYS, values)}
    )


class TestLaminateLoad(unittest.TestCase):
    def test_vector(self):
        laminate_load = MyOnlineExtensionExerciseTestLoad.instantiate()
        expected_value = np.zeros((6, 1))
        expected_value[0, 0] = laminate_load.Nx
        actual_value = laminate_load.vector
        np.testing.assert_array_equal(expected_value, actual_value)

    @given(
        my_load_strategy(), st.decimals(allow_nan=False, allow_infinity=False)
    )
    def test_scale(self, load: LaminateLoad, factor: float):
        factor = float(factor)

        scaled_values = {key: getattr(load, key) * factor for key in LOAD_KEYS}

        expected_value = load.scale(factor)
        actual_value = LaminateLoad(**scaled_values)

        self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
    unittest.main()


# End
