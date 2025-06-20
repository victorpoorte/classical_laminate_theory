from typing import Protocol
import numpy as np


class VesselLayer(Protocol):
    def compute_global_strain_vector(
        self,
        radius: float,
        layer_index: int,
        solution: np.ndarray,
        temperature_delta: float,
    ) -> np.ndarray:
        ...


class LaminateLayer(Protocol):
    def compute_local_strain(self, global_strain: np.ndarray) -> np.ndarray:
        ...


class Vessel(Protocol):
    inner_radius: float
    outer_radius: float

    def get_layers(
        self, radius: float
    ) -> tuple[int, VesselLayer, LaminateLayer]:
        ...


def compute_through_thickness_strains(
    solution: np.ndarray, vessel: Vessel, temperature_delta: float, laminate_layers: list[LaminateLayer]
) -> None:
    radii = np.linspace(vessel.inner_radius, vessel.outer_radius, num=int(1e3))

    local_strains = np.zeros((6, len(radii)))
    for j, radius in enumerate(radii):
        i, tube_layer = vessel.get_layers(radius)
        laminate_layer = laminate_layers[i]
        global_strain = tube_layer.compute_global_strain_vector(
            radius, i, solution, temperature_delta
        )
        local_strains[:, j] = laminate_layer.compute_local_strain(
            global_strain
        )

    x_value = (radii - vessel.inner_radius) / (
        vessel.outer_radius - vessel.inner_radius
    )
    return local_strains, x_value


def main():
    pass


if __name__ == "__main__":
    main()


# End
