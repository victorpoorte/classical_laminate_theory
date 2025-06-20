from abc import abstractmethod
from typing import Protocol
import numpy as np


from ..tube_layers import TubeLayer

from ..tube import Tube


class TubeLayer(TubeLayer):
    beta: float
    alpha: float

    @abstractmethod
    def elongation_stress_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def load_vector_stress_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def elongation_displacement_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def load_vector_displacement_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def elongation_elongation_entry(
        self, radius1: float, radius2: float
    ) -> float:
        ...

    @abstractmethod
    def load_vector_elongation_entry(
        self, radius1: float, radius2: float
    ) -> float:
        ...


class VesselLoad(Protocol):
    temperature_delta: float
    inner_pressure: float
    outer_pressure: float
    pressure_difference: float
    axial_load: float


class Tube(Tube):
    layers: list[TubeLayer]
    first_layer: TubeLayer
    last_layer: TubeLayer

    @property
    def no_of_equations(self):
        return self.no_of_layers * 2 + 1

    def displacement_row(
        self, layer1: TubeLayer, layer2: TubeLayer, i: int
    ) -> np.ndarray:
        row = np.zeros(self.no_of_equations)
        radius = self.radii[i + 1]
        row[i] = layer1.first_displacement_entry(radius)
        row[i + 1] = -layer2.first_displacement_entry(radius)
        row[i + self.no_of_layers] = layer1.second_displacement_entry(radius)
        row[i + self.no_of_layers + 1] = -layer2.second_displacement_entry(
            radius
        )
        row[-1] = layer2.elongation_displacement_entry(
            radius
        ) - layer1.elongation_displacement_entry(radius)
        return row

    def stress_row(
        self, layer1: TubeLayer, layer2: TubeLayer, i: int
    ) -> np.ndarray:
        row = np.zeros(self.no_of_equations)
        radius = self.radii[i + 1]
        row[i] = layer1.first_stress_entry(radius)
        row[i + 1] = -layer2.first_stress_entry(radius)
        row[i + self.no_of_layers] = layer1.second_stress_entry(radius)
        row[i + self.no_of_layers + 1] = -layer2.second_stress_entry(radius)
        row[-1] = layer2.elongation_stress_entry(
            radius
        ) - layer1.elongation_stress_entry(radius)
        return row

    def internal_stress_row(self) -> np.ndarray:
        row = np.zeros((1, self.no_of_equations))
        row[0, 0] = self.first_layer.first_stress_entry(self.inner_radius)
        row[0, self.no_of_layers] = self.first_layer.second_stress_entry(
            self.inner_radius
        )
        row[0, -1] = self.first_layer.elongation_stress_entry(
            self.inner_radius
        )
        return row

    def external_stress_row(self) -> np.ndarray:
        row = np.zeros(self.no_of_equations)
        row[self.no_of_layers - 1] = self.last_layer.first_stress_entry(
            self.outer_radius
        )
        row[-2] = self.last_layer.second_stress_entry(self.outer_radius)
        row[-1] = self.last_layer.elongation_stress_entry(self.outer_radius)
        return row

    def elongation_row(self) -> np.ndarray:
        row = np.zeros(self.no_of_equations)
        for i, layer in enumerate(self.layers):
            radius1 = self.radii[i]
            radius2 = self.radii[i + 1]
            row[i] = layer.first_elongation_entry(radius1, radius2)
            row[i + self.no_of_layers] = layer.second_elongation_entry(
                radius1, radius2
            )
            row[-1] += layer.elongation_elongation_entry(radius1, radius2)
        return row

    def create_loading_vector(
        self,
        load: VesselLoad,
    ) -> np.ndarray:
        # Set up the empty vector to be filled
        vector = np.zeros((self.no_of_equations, 1))

        # Inner layer stress compatibility
        vector[0, 0] = (
            -load.inner_pressure
            - self.first_layer.load_vector_stress_entry(self.inner_radius)
            * load.temperature_delta
        )

        # Strain and stress compatibility between the layers
        for i, _ in enumerate(self.layers[:-1]):
            layer1 = self.layers[i]
            layer2 = self.layers[i + 1]
            radius = self.radii[i + 1]
            vector[i + 1] = (
                layer2.load_vector_displacement_entry(radius)
                - layer1.load_vector_displacement_entry(radius)
            ) * load.temperature_delta
            vector[i + self.no_of_layers] = (
                layer2.load_vector_stress_entry(radius)
                - layer1.load_vector_stress_entry(radius)
            ) * load.temperature_delta

        # Outer stress compatibility
        vector[-2, 0] = (
            -load.outer_pressure
            - self.last_layer.load_vector_stress_entry(radius)
            * load.temperature_delta
        )

        # Extension of the vessel
        vector[-1, 0] += self.inner_radius**2 * load.pressure_difference
        vector[-1, 0] += load.axial_load / np.pi
        for i, layer in enumerate(self.layers):
            vector[-1, 0] -= (
                layer.load_vector_elongation_entry(
                    self.radii[i], self.radii[i + 1]
                )
                * load.temperature_delta
            )

        return vector

    def longitudinal_compatibility(self) -> np.ndarray:
        return self.elongation_row()

    def create_force_derivative_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.no_of_equations, self.no_of_equations))
        matrix[-1, 0] = self.inner_radius ** (self.first_layer.beta - 1)
        matrix[-1, self.no_of_layers] = self.inner_radius ** (
            -self.first_layer.beta - 1
        )
        matrix[-1, -1] = self.first_layer.alpha / (
            1 - self.first_layer.beta**2
        )

        return matrix


def main():
    pass


if __name__ == "__main__":
    main()


# End
