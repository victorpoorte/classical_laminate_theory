import numpy as np

from typing import Protocol

from classical_laminate_theory.clt.failure_criteria.puck import FailureIndexResult


from .failure_strategy import FailureResult, FailureStrategy


class Layer(Protocol):
    ...

class Load(Protocol):
    inner_pressure: float
    temperature_delta: float
    operating_temperature: float


class StrainComputer(Protocol):
    def compute_global_strains(self, layers: list[Layer], load: Load) -> np.ndarray:
        ...


class FailureCriterion(Protocol):
    def long_and_trav_index(self, strain: np.ndarray, layer: Layer, temperature_delta: float, operating_temperature: float) -> tuple[float, float]:
        ...

    def factor_of_safety(
        self, global_strain: np.ndarray, layer: Layer, temperature_delta: float, operating_temperature: float
    ) -> float:
        ...


class LayeringStrategy(Protocol):
    ...


class FailureAnalyser:
    
    def __init__(self, strain_computer: StrainComputer, failure_criterion: FailureCriterion, failure_strategy: FailureStrategy) -> None:
        self.strain_computer = strain_computer
        self.failure_criterion = failure_criterion
        self.failure_strategy = failure_strategy

    def analyse(self, layers: list[Layer], load: Load) -> FailureResult:
        return self.failure_strategy.run(layers, load, self)

    def compute_critical_factor(self, layers: list[Layer], load: Load) -> float:

        global_strains = self.strain_computer.compute_global_strains(layers, load)
        
        # Compute failure indices, and extract the critical ones
        factors = [
            self.failure_criterion.factor_of_safety(
                strain, layer, load.temperature_delta, load.operating_temperature
            )
            for layer, strain in zip(layers, global_strains.transpose())
        ]

        factors, modes = zip(*factors)
        min_index = np.argmin(factors)
        min_factor = factors[min_index]

        return min_factor, modes[min_index]

    def compute_failure_indices(self, layers: list[Layer], load: Load):
        # Compute global strains
        global_strains = self.strain_computer.compute_global_strains(layers, load)

        # Compute failure indices, and extract the critical ones
        return [
            self.failure_criterion.long_and_trav_index(
                strain, layer, load.temperature_delta, load.operating_temperature
            )
            for layer, strain in zip(layers, global_strains.transpose())
        ]

    @classmethod
    def _extract_critical_failure_index(cls, indices: list[FailureIndexResult]) -> float:
        return max(indices, key=lambda x: max(x.longitudinal, x.transverse)).critical

    @classmethod
    def _determine_critical_index_location(cls, indices: list[FailureIndexResult]) -> int:
        return indices.index(max(indices, key=lambda x: max(x.longitudinal, x.transverse)))



def main():
    pass


if __name__ == "__main__":
    main()


# End