import math
from dataclasses import dataclass
from functools import cached_property
import numpy as np

from ..tube_layers import TubeLayer


@dataclass
class TubeLayer(TubeLayer):
    thickness: float
    C11: float
    C22: float
    C33: float
    C12: float
    C13: float
    C23: float
    C16: float
    C26: float
    C36: float
    C66: float

    @cached_property
    def alpha_1(self):
        num = self.C12 - self.C13
        if num == 0:
            return 0
        return num / (self.C33 - self.C22)

    @cached_property
    def alpha_2(self):
        return (self.C26 - 2 * self.C36) / (4 * self.C33 - self.C22)

    @cached_property
    def beta(self):
        return math.sqrt(self.C22 / self.C33)

    def first_stress_entry(self, radius: float) -> float:
        return (self.C23 + self.beta * self.C33) * radius ** (self.beta - 1)

    def second_stress_entry(self, radius: float) -> float:
        return (self.C23 - self.beta * self.C33) * radius ** (-self.beta - 1)

    def elongation_stress_entry(self, _: float) -> float:
        return self.C13 + self.alpha_1 * (self.C23 + self.C33)

    def twist_stress_entry(self, radius: float) -> float:
        return ((self.C36) + self.alpha_2 * (self.C23 + 2 * self.C33)) * radius

    def first_displacement_entry(self, radius: float) -> float:
        return radius**self.beta

    def second_displacement_entry(self, radius: float) -> float:
        return radius ** (-self.beta)

    def elongation_displacement_entry(self, radius: float) -> float:
        return self.alpha_1 * radius

    def twist_displacement_entry(self, radius: float) -> float:
        return self.alpha_2 * radius**2

    def first_elongation_entry(
        self, radius_1: float, radius_2: float
    ) -> float:
        factor_1 = (self.C12 + self.beta * self.C13) / (1 + self.beta)
        factor_2 = radius_2 ** (self.beta + 1) - radius_1 ** (self.beta + 1)
        return factor_1 * factor_2

    def second_elongation_entry(
        self, radius_1: float, radius_2: float
    ) -> float:
        if self.beta == 1:
            return 0
        factor_1 = (self.C12 - self.beta * self.C13) / (1 - self.beta)
        factor_2 = radius_2 ** (-self.beta + 1) - radius_1 ** (-self.beta + 1)
        return factor_1 * factor_2

    def elongation_elongation_entry(
        self, radius_1: float, radius_2: float
    ) -> float:
        return (self.C11 + self.alpha_1 * (self.C12 + self.C13)) * (
            (radius_2**2 - radius_1**2) / 2
        )

    def twist_elongation_entry(
        self, radius_1: float, radius_2: float
    ) -> float:
        return (
            (self.C16 + self.alpha_2 * (self.C12 + 2 * self.C13))
            * (radius_2**3 - radius_1**3)
            / 3
        )

    def first_twist_entry(self, radius_1: float, radius_2: float) -> float:
        numerator = self.C26 + self.beta * self.C36
        denominator = 2 + self.beta
        factor_2 = radius_2 ** (self.beta + 2) - radius_1 ** (self.beta + 2)

        return numerator / denominator * factor_2

    def second_twist_entry(self, radius_1: float, radius_2: float) -> float:
        numerator = self.C26 - self.beta * self.C36
        denominator = 2 - self.beta
        factor_2 = radius_2 ** (-self.beta + 2) - radius_1 ** (-self.beta + 2)

        return numerator / denominator * factor_2

    def elongation_twist_entry(
        self, radius_1: float, radius_2: float
    ) -> float:
        return (
            (self.C16 + self.alpha_1 * (self.C26 + self.C36))
            * (radius_2**3 - radius_1**3)
            / 3
        )

    def twist_twist_entry(self, radius_1: float, radius_2: float) -> float:
        return (
            (self.C66 + self.alpha_2 * (self.C26 + 2 * self.C36))
            * (radius_2**4 - radius_1**4)
            / 4
        )

    def u(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        _: float,
    ) -> float:
        d, e, epsilon_0, gamma_0 = self._unpack_solution_vector(
            solution_vector, layer_index
        )
        return (
            d * radius**self.beta
            + e * radius ** (-self.beta)
            + self.alpha_1 * epsilon_0 * radius
            + self.alpha_2 * gamma_0 * radius**2
        )

    def du_dr(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        _: float,
    ) -> float:
        d, e, epsilon_0, gamma_0 = self._unpack_solution_vector(
            solution_vector, layer_index
        )
        return (
            d * self.beta * radius ** (self.beta - 1)
            - e * self.beta * radius ** (-self.beta - 1)
            + self.alpha_1 * epsilon_0
            + 2 * self.alpha_2 * gamma_0 * radius
        )

    @classmethod
    def _unpack_solution_vector(
        cls,
        solution_vector: np.ndarray,
        layer_index: int,
    ) -> tuple[float, float, float, float]:
        number_of_layers = cls.number_of_layers(solution_vector)
        d = cls._unpack_D_from_solution_vector(
            solution_vector,
            layer_index,
            number_of_layers,
        )
        e = cls._unpack_E_from_solution_vector(
            solution_vector, layer_index, number_of_layers
        )
        epsilon_0 = cls._unpack_epsilon_from_solution_vector(
            solution_vector, layer_index, number_of_layers
        )
        gamma_0 = cls._unpack_gamma_from_solution_vector(
            solution_vector, layer_index, number_of_layers
        )

        return d, e, epsilon_0, gamma_0

    @staticmethod
    def _unpack_D_from_solution_vector(
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> float:
        return solution_vector[layer_index, 0]

    @staticmethod
    def _unpack_E_from_solution_vector(
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> float:
        return solution_vector[layer_index + number_of_layers, 0]

    @staticmethod
    def _unpack_epsilon_from_solution_vector(
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> float:
        return solution_vector[-2, 0]

    @staticmethod
    def _unpack_gamma_from_solution_vector(
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> float:
        return solution_vector[-1, 0]

    @staticmethod
    def number_of_layers(solution_vector: np.ndarray) -> int:
        return int((len(solution_vector) - 2) / 2)


def main():
    pass


if __name__ == "__main__":
    main()


# End
