import numpy as np

from typing import Protocol

from ...computer_protocol import PressureVesselStrainComputer


class Load(Protocol):
    temperature_delta: float


class TubeLayer(Protocol):
    def compute_global_strain_vector(
        self, radius: float, solution_vector: np.ndarray, temperature: float
    ) -> np.ndarray:
        ...


class Vessel(Protocol):
    layers: list[TubeLayer]
    mid_layer_radii: list[float]
    no_of_layers: int

#     def compute_solution_vector(self, load: Load) -> np.ndarray:
#         ...


# class Lamina(Protocol):
#     ...


# class LaminateLayer(Protocol):
#     ...



class VesselFactory(Protocol):
    ...
    # def create_vessel(
    #     self,
    #     inner_radius: float,
    #     orientations: list[float],
    #     thicknesses: list[float],
    #     materials: list[Lamina],
    #     degrees: bool,
    # ) -> Vessel:
    #     ...

    # def create_laminate_layers(
    #     self,
    #     orientations: list[float],
    #     thicknesses: list[float],
    #     materials: list[Lamina],
    #     degrees: bool,
    # ) -> list[LaminateLayer]:
    #     ...



class ThickWalledStrainComputer(PressureVesselStrainComputer):
    def __init__(
        self,
        vessel_factory: VesselFactory,
    ) -> None:
        self.vessel_factory = vessel_factory

    def _compute_global_strains(
        self,
        load: Load,
        tube: Vessel,
        solution_vector: np.ndarray,
    ) -> np.ndarray:
        strains = np.empty((6, tube.no_of_layers))
        for i, layer in enumerate(tube.layers):
            strains[:, i] = layer.compute_global_strain_vector(
                tube.mid_layer_radii[i],
                i,
                solution_vector,
                load.temperature_delta,
            )

        return strains


def main():
    pass


if __name__ == "__main__":
    main()


# End
