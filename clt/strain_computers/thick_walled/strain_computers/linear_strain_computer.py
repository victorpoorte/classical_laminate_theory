from typing import Protocol

import numpy as np

from .thick_tube_strain_computer_protocol import ThickWalledStrainComputer
from ..tubes.general_loaded.tube_factory import VesselFactory


class Lamina(Protocol):
    ...


class Load(Protocol):
    ...


class Vessel(Protocol):
    def compute_solution_vector(self, load: Load) -> np.ndarray:
        ...


class LaminateLayer(Protocol):
    ...


class Layer(Protocol):
    ...


class LinearThickWalledStrainComputer(ThickWalledStrainComputer):

    def __init__(self) -> None:
        super().__init__(VesselFactory())

    def __str__(self) -> str:
        return "Linear Thick-Walled"

    def compute_solution_vector(
        self, layers: list[Layer], load: Load, inner_radius: float
    ) -> np.ndarray:
        tube: Vessel = self.vessel_factory.create_vessel(
            inner_radius, layers
        )
        return tube.compute_solution_vector(load)

    def compute_global_strains(
        self, layers: list[Layer], load: Load, inner_radius: float
    ) -> np.ndarray:
        tube: Vessel = self.vessel_factory.create_vessel(inner_radius, layers)
        solution_vector = tube.compute_solution_vector(load)
        global_strains = self._compute_global_strains(
            load, tube, solution_vector
        )
        return global_strains


def main():
    pass


if __name__ == "__main__":
    main()


# End
