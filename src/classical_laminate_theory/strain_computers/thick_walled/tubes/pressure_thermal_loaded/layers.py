import math
from abc import abstractmethod
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
    xi: np.ndarray

    @cached_property
    def alpha(self):
        return (self.C12 - self.C13) / self.C33

    @cached_property
    def beta(self):
        return math.sqrt(abs(self.C22 / self.C33))

    @property
    def xi_r(self):
        return self.xi[2, 0]

    @property
    def xi_theta(self):
        return self.xi[1, 0]

    @property
    def xi_z(self):
        return self.xi[0, 0]

    @property
    def eta(self):
        return (self.xi_r - self.xi_theta) / self.C33

    @abstractmethod
    def load_factor(self, radius: float) -> float:
        ...

    @abstractmethod
    def compliance_factor(self, radius: float) -> float:
        ...

    def elongation_stress_entry(self, radius: float) -> float:
        return (
            self.compliance_factor(radius) * (self.C33 + self.C23) + self.C13
        )

    def load_vector_stress_entry(self, radius: float) -> float:
        return self.load_factor(radius) * (self.C33 + self.C23) - self.xi_r

    def elongation_displacement_entry(self, radius: float) -> float:
        return self.compliance_factor(radius) * radius

    def load_vector_displacement_entry(self, radius: float) -> float:
        return self.load_factor(radius) * radius

    def elongation_elongation_entry(
        self, radius1: float, radius2: float
    ) -> float:
        return (
            self.compliance_factor(radius1) * (self.C13 + self.C12) + self.C11
        ) * (radius2**2 - radius1**2)

    def load_vector_elongation_entry(
        self, radius1: float, radius2: float
    ) -> float:
        return (
            self.load_factor(radius1) * (self.C13 + self.C12) - self.xi_z
        ) * (radius2**2 - radius1**2)

    @staticmethod
    def _unpack_epsilon_from_solution_vector(
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> float:
        return solution_vector[-1, 0]

    @staticmethod
    def _unpack_gamma_from_solution_vector(
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> float:
        return 0

    @classmethod
    def _unpack_solution_vector(
        cls,
        solution_vector: np.ndarray,
        layer_index: int,
    ) -> tuple[float, float, float, float]:
        number_of_layers = cls.number_of_layers(solution_vector)
        a = cls._unpack_A_from_solution_vector(
            solution_vector,
            layer_index,
            number_of_layers,
        )
        b = cls._unpack_B_from_solution_vector(
            solution_vector, layer_index, number_of_layers
        )
        epsilon_0 = cls._unpack_epsilon_from_solution_vector(
            solution_vector, layer_index, number_of_layers
        )
        gamma_0 = cls._unpack_gamma_from_solution_vector(
            solution_vector, layer_index, number_of_layers
        )

        return a, b, epsilon_0, gamma_0

    @staticmethod
    def _unpack_A_from_solution_vector(
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> float:
        return solution_vector[layer_index, 0]

    @staticmethod
    def _unpack_B_from_solution_vector(
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> float:
        return solution_vector[layer_index + number_of_layers, 0]

    @staticmethod
    def number_of_layers(solution_vector: np.ndarray) -> int:
        return int((len(solution_vector) - 1) / 2)


class Case1Layer(TubeLayer):
    # C22/C33 > 0

    def compliance_factor(self, _: float) -> float:
        return self.alpha / (1 - self.beta**2)

    def load_factor(self, _: float) -> float:
        return self.eta / (1 - self.beta**2)

    def first_stress_entry(self, radius: float) -> float:
        return (self.beta * self.C33 + self.C23) * radius ** (self.beta - 1)

    def second_stress_entry(self, radius: float) -> float:
        return (-self.beta * self.C33 + self.C23) * radius ** (-self.beta - 1)

    def first_displacement_entry(self, radius: float) -> float:
        return radius**self.beta

    def second_displacement_entry(self, radius: float) -> float:
        return radius ** (-self.beta)

    def first_elongation_entry(self, radius1: float, radius2: float) -> float:
        return (
            2
            * (self.beta * self.C13 + self.C12)
            / (1 + self.beta)
            * (radius2 ** (self.beta + 1) - radius1 ** (self.beta + 1))
        )

    def second_elongation_entry(self, radius1: float, radius2: float) -> float:
        return (
            2
            * (-self.beta * self.C13 + self.C12)
            / (1 - self.beta)
            * (radius2 ** (-self.beta + 1) - radius1 ** (-self.beta + 1))
        )

    def u(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        temperature_delta: float,
    ) -> float:
        a, b, epsilon_0, _ = self._unpack_solution_vector(
            solution_vector, layer_index
        )
        return (
            a * radius ** (self.beta)
            + b * radius ** (-self.beta)
            + (self.alpha * epsilon_0 + self.eta * temperature_delta)
            * radius
            / (1 - self.beta**2)
        )

    def du_dr(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        temperature_delta: float,
    ) -> float:
        a, b, epsilon_0, _ = self._unpack_solution_vector(
            solution_vector, layer_index
        )
        return (
            self.beta * a * radius ** (self.beta - 1)
            - self.beta * b * radius ** (-self.beta - 1)
            + (self.alpha * epsilon_0 + self.eta * temperature_delta)
            / (1 - self.beta**2)
        )


class Case2Layer(TubeLayer):
    # C22/C33 < 0
    def compliance_factor(self, _: float):
        return self.alpha / (1 + self.beta**2)

    def load_factor(self, _: float):
        return self.eta / (1 + self.beta**2)

    def first_stress_entry(self, radius: float) -> float:
        return (
            -self.beta * self.C33 * math.sin(self.beta * math.log(radius))
            + self.C23 * math.cos(self.beta * math.log(radius))
        ) / radius

    def second_stress_entry(self, radius: float) -> float:
        return (
            self.beta * self.C33 * math.cos(self.beta * math.log(radius))
            + self.C23 * math.sin(self.beta * math.log(radius))
        ) / radius

    def first_displacement_entry(self, radius: float) -> float:
        return math.cos(self.beta * math.log(radius))

    def second_displacement_entry(self, radius: float) -> float:
        return math.sin(self.beta * math.log(radius))

    def first_elongation_entry(self, radius1: float, radius2: float) -> float:
        return (
            2
            / (1 + self.beta**2)
            * (
                self.beta
                * (self.C12 - self.C13)
                * (
                    radius2 * math.sin(self.beta * math.log(radius2))
                    - radius1 * math.sin(self.beta * math.log(radius1))
                )
                + (self.C13 * self.beta**2 + self.C12)
                * (
                    radius2 * math.cos(self.beta * math.log(radius2))
                    - radius1 * math.cos(self.beta * math.log(radius1))
                )
            )
        )

    def second_elongation_entry(self, radius1: float, radius2: float) -> float:
        return (
            2
            / (1 + self.beta**2)
            * (
                (self.C13 * self.beta**2 + self.C12)
                * (
                    radius2 * math.sin(self.beta * math.log(radius2))
                    - radius1 * math.sin(self.beta * math.log(radius1))
                )
                + self.beta
                * (self.C13 - self.C12)
                * (
                    radius2 * math.cos(self.beta * math.log(radius2))
                    - radius1 * math.cos(self.beta * math.log(radius1))
                )
            )
        )

    def u(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        temperature_delta: float,
    ) -> float:
        a, b, epsilon_0, _ = self._unpack_solution_vector(
            solution_vector, layer_index
        )
        return (
            a * math.cos(self.beta * math.log(radius))
            + b * math.sin(self.beta * math.log(radius))
            + (self.alpha * epsilon_0 + self.eta * temperature_delta)
            * radius
            / (1 + self.beta**2)
        )

    def du_dr(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        temperature_delta: float,
    ) -> float:
        a, b, epsilon_0, _ = self._unpack_solution_vector(
            solution_vector, layer_index
        )
        return (
            -a * self.beta * math.sin(self.beta * math.log(radius)) / radius
            + b * self.beta * math.cos(self.beta * math.log(radius)) / radius
            + (self.alpha * epsilon_0 + self.eta * temperature_delta)
            / (1 + self.beta**2)
        )


class Case3Layer(Case1Layer):
    def compliance_factor(self, radius: float):
        return self.alpha * math.log(radius) / 2

    def load_factor(self, radius: float):
        return self.eta * math.log(radius) / 2

    def elongation_stress_entry(self, radius: float):
        return 0.5 * (
            self.alpha * math.log(radius) * (self.C33 + self.C23)
            + self.alpha * self.C33
            + 2 * self.C13
        )

    def second_elongation_entry(self, radius1: float, radius2: float) -> float:
        return 2 * (self.C12 - self.C13) * math.log(radius2 / radius1)

    def elongation_elongation_entry(
        self, radius1: float, radius2: float
    ) -> float:
        return self.alpha * (self.C13 + self.C12) * (
            (
                radius2**2 * math.log(radius2)
                - radius1**2 * math.log(radius1)
            )
            / 2
            - (radius2**2 - radius1**2) / 4
        ) + (self.alpha * self.C13 / 2 + self.C11) * (
            radius2**2 - radius1**2
        )

    def load_vector_stress_entry(self, radius: float):
        return (
            self.eta * math.log(radius) * (self.C33 + self.C23)
            + self.eta * self.C33
        ) / 2 - self.xi_r

    def load_vector_elongation_entry(self, radius1: float, radius2: float):
        return self.eta * (self.C13 + self.C12) * (
            (
                radius2**2 * math.log(radius2)
                - radius1**2 * math.log(radius1)
            )
            / 2
            - (radius2**2 - radius1**2) / 4
        ) + (self.eta * self.C13 / 2 - self.xi_z) * (
            radius2**2 - radius1**2
        )

    def u(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        temperature_delta: float,
    ) -> float:
        a, b, epsilon_0, _ = self._unpack_solution_vector(
            solution_vector, layer_index
        )
        return (
            (self.alpha * epsilon_0 + self.eta * temperature_delta)
            * radius
            * math.log(radius)
            / 2
            + a * radius
            + b / radius
        )

    def du_dr(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        temperature_delta: float,
    ) -> float:
        a, b, epsilon_0, _ = self._unpack_solution_vector(
            solution_vector, layer_index
        )
        return (
            (self.alpha * epsilon_0 + self.eta * temperature_delta)
            * (math.log(radius) + 1)
            / 2
            + a
            - b / (radius**2)
        )


def main():
    pass


if __name__ == "__main__":
    main()


# End
