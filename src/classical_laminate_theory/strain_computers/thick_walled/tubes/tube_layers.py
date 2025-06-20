from abc import abstractmethod
from typing import Protocol

import numpy as np


class TubeLayer(Protocol):
    @abstractmethod
    def first_stress_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def second_stress_entry(self, radius: float) -> float:
        ...

    # @abstractmethod
    # def elongation_stress_entry(self, radius: float) -> float:
    #     ...

    @abstractmethod
    def first_displacement_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def second_displacement_entry(self, radius: float) -> float:
        ...

    # @abstractmethod
    # def elongation_displacement_entry(self, radius: float) -> float:
    #     ...

    @abstractmethod
    def first_elongation_entry(self, radius1: float, radius2: float) -> float:
        ...

    @abstractmethod
    def second_elongation_entry(self, radius1: float, radius2: float) -> float:
        ...

    # @abstractmethod
    # def elongation_elongation_entry(
    #     self, radius1: float, radius2: float
    # ) -> float:
    #     ...

    @abstractmethod
    def u(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        temperature_delta: float,
    ) -> float:
        ...

    @abstractmethod
    def du_dr(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        temperature_delta: float,
    ) -> float:
        ...

    @abstractmethod
    def number_of_layers(solution_vector: np.ndarray) -> int:
        ...

    def compute_global_strain_vector(
        self,
        radius: float,
        index: int,
        solution_vector: np.ndarray,
        temperature_delta: float,
    ) -> np.ndarray:
        no_of_layers = self.number_of_layers(solution_vector)
        u = self.u(
            radius,
            solution_vector,
            index,
            no_of_layers,
            temperature_delta,
        )
        du_dr = self.du_dr(
            radius,
            solution_vector,
            index,
            no_of_layers,
            temperature_delta,
        )
        gamma_0 = self._unpack_gamma_from_solution_vector(
            solution_vector, index, no_of_layers
        )
        epsilon_0 = self._unpack_epsilon_from_solution_vector(
            solution_vector, index, no_of_layers
        )
        return np.array(
            [
                epsilon_0,
                u / radius,
                du_dr,
                0,
                0,
                gamma_0 * radius,
            ]
        )

    @abstractmethod
    def _unpack_epsilon_from_solution_vector(
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> float:
        ...

    @abstractmethod
    def _unpack_gamma_from_solution_vector(
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> float:
        ...


def main():
    pass


if __name__ == "__main__":
    main()


# End
