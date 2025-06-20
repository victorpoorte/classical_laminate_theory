from __future__ import annotations
from abc import abstractmethod

from dataclasses import dataclass
from typing import Protocol
import numpy as np

from .strain_computers.non_linear_strain_computer import NonLinearStep


class Layer(Protocol):
    ...


class LaminateLayer(Protocol):
    ...


class Load(Protocol):
    temperature_delta: float

    def determine_scaling_factor(self, load: Load) -> float:
        ...

    def scale(self, factor: float) -> Load:
        ...


class FailureCriterion(Protocol):
    def factor_of_safety(
        self, global_strain: np.ndarray, layer: Layer, temperature_delta: float
    ) -> float:
        ...


class VesselFactory(Protocol):
    def determine_number_of_equations(self, number_of_layers: int) -> int:
        ...


class StrainComputer(Protocol):
    inner_radius: float
    vessel_factory: VesselFactory

    def compute_solution_vector(
        self, layers: list[Layer], load: Load
    ) -> np.ndarray:
        ...

    def compute_strain_from_solution_vector(
        self, layers: list[Layer], load: Load, solution: np.ndarray
    ) -> np.ndarray:
        ...


class CriticalFactorComputer(Protocol):
    @abstractmethod
    def determine_critical_factor(
        self,
        layers: list[Layer],
        load: Load,
        failure_criterion: FailureCriterion,
        strain_computer: StrainComputer,
    ) -> float:
        ...

    def _compute_critical_factor(
        self,
        load: Load,
        failure_criterion: FailureCriterion,
        strains: np.ndarray,
        laminate_layers: list[LaminateLayer],
    ) -> float:
        factor = min(
            failure_criterion.factor_of_safety(
                strains[:, i], layer, load.temperature_delta
            )
            for i, layer in enumerate(laminate_layers)
        )

        return factor


@dataclass
class NonLinearVesselFactorComputer(CriticalFactorComputer):
    non_linear_method: callable
    tolerance: float = 1e-2
    max_iterations: int = 1e3

    def determine_critical_factor(
        self,
        layers: list[Layer],
        load: Load,
        failure_criterion: FailureCriterion,
        strain_computer: StrainComputer,
    ) -> float:
        original_load = load
        previous_solution = np.zeros(
            (
                strain_computer.vessel_factory.determine_number_of_equations(
                    len(layers)
                ),
                1,
            )
        )
        for _ in range(int(self.max_iterations)):
            step = NonLinearStep(
                strain_computer.inner_radius,
                layers,
                strain_computer.vessel_factory,
                load,
                self.non_linear_method,
            )
            solution = step.compute_solution_vector(previous_solution)
            (
                strains,
                lam_layers,
            ) = strain_computer.compute_strain_from_solution_vector(
                layers, load, solution
            )
            critical_factor = self._compute_critical_factor(
                load, failure_criterion, strains, lam_layers
            )
            if abs(1 - critical_factor) <= self.tolerance:
                return load.determine_scaling_factor(original_load)
            if critical_factor > 1:
                previous_solution = solution
            load = load.scale(critical_factor)
        raise StopIteration("Max number of iterations reached...")


def main():
    pass


if __name__ == "__main__":
    main()


# End
