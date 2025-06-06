from __future__ import annotations

import unittest
from test.test_strategies import my_list_of_numbers_strategy

import hypothesis.strategies as st

from clt.material import FailureStresses, Lamina, LaminaFailureStresses


@st.composite
def my_failure_stresses_strategy(draw):
    failure_variables = 3
    variables = draw(my_list_of_numbers_strategy(failure_variables))
    variables[0] *= -1
    return FailureStresses(*variables)


@st.composite
def my_laminate_failure_stresses_strategy(draw):
    return LaminaFailureStresses(
        draw((my_failure_stresses_strategy())),
        draw(my_failure_stresses_strategy()),
    )


@st.composite
def my_lamina_strategy(draw):
    lamina_variables = 4
    return Lamina(
        *draw(my_list_of_numbers_strategy(lamina_variables)),
        failure_stresses=draw(my_laminate_failure_stresses_strategy()),
    )


if __name__ == "__main__":
    unittest.main()


# End
