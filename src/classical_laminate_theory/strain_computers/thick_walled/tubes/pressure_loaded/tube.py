from abc import abstractmethod
from typing import Protocol
import numpy as np


from ..tube import Tube
from ..tube_layers import TubeLayer


class VesselLoad(Protocol):
    inner_pressure: float
    outer_pressure: float
    pressure_difference: float


class TubeLayer(TubeLayer):
    beta: float
    alpha_1: float
    alpha_2: float

    @abstractmethod
    def elongation_displacement_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def twist_displacement_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def elongation_stress_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def twist_stress_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def elongation_elongation_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def twist_elongation_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def first_twist_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def second_twist_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def elongation_twist_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def twist_twist_entry(self, radius: float) -> float:
        ...


class Tube(Tube):
    layers: list[TubeLayer]
    first_layer: TubeLayer
    last_layer: TubeLayer

    @property
    def no_of_equations(self):
        return self.no_of_layers * 2 + 2

    def displacement_row(
        self, layer1: TubeLayer, layer2: TubeLayer, i: int
    ) -> list:
        row = np.zeros(self.no_of_equations)
        radius = self.radii[i + 1]
        row[i] = layer1.first_displacement_entry(radius)
        row[i + 1] = -layer2.first_displacement_entry(radius)
        row[i + self.no_of_layers] = layer1.second_displacement_entry(radius)
        row[i + self.no_of_layers + 1] = -layer2.second_displacement_entry(
            radius
        )
        row[-2] = layer1.elongation_displacement_entry(
            radius
        ) - layer2.elongation_displacement_entry(radius)
        row[-1] = layer1.twist_displacement_entry(
            radius
        ) - layer2.twist_displacement_entry(radius)

        return row

    def stress_row(self, layer1: TubeLayer, layer2: TubeLayer, i: int) -> list:
        row = np.zeros(self.no_of_equations)
        radius = self.radii[i + 1]
        row[i] = layer1.first_stress_entry(radius)
        row[i + 1] = -layer2.first_stress_entry(radius)
        row[i + self.no_of_layers] = layer1.second_stress_entry(radius)
        row[i + self.no_of_layers + 1] = -layer2.second_stress_entry(radius)
        row[-2] = layer1.elongation_stress_entry(
            radius
        ) - layer2.elongation_stress_entry(radius)
        row[-1] = layer1.twist_stress_entry(
            radius
        ) - layer2.twist_stress_entry(radius)

        return row

    def external_stress_row(self):
        row = np.zeros(self.no_of_equations)
        row[self.no_of_layers - 1] = self.last_layer.first_stress_entry(
            self.outer_radius
        )
        row[-3] = self.last_layer.second_stress_entry(self.outer_radius)
        row[-2] = self.last_layer.elongation_stress_entry(self.outer_radius)
        row[-1] = self.last_layer.twist_stress_entry(self.outer_radius)
        return row

    def internal_stress_row(self) -> np.ndarray:
        row = np.zeros((1, self.no_of_equations))
        row[0, 0] = self.first_layer.first_stress_entry(self.inner_radius)
        row[0, self.no_of_layers] = self.first_layer.second_stress_entry(
            self.inner_radius
        )
        row[0, -2] = self.first_layer.elongation_stress_entry(
            self.inner_radius
        )
        row[0, -1] = self.first_layer.twist_stress_entry(self.inner_radius)
        return row

    def longitudinal_compatibility(self) -> np.ndarray:
        matrix = np.zeros((2, self.no_of_equations))
        # Fill the last rows of the matrix
        for i, layer in enumerate(self.layers):
            # Extract the inner and outer radii of the layer
            radius_1 = self.radii[i]
            radius_2 = self.radii[i + 1]

            # Fill the elongation entries
            matrix[0, i] = layer.first_elongation_entry(radius_1, radius_2)
            matrix[0, i + self.no_of_layers] = layer.second_elongation_entry(
                radius_1, radius_2
            )
            matrix[0, -2] += layer.elongation_elongation_entry(
                radius_1, radius_2
            )
            matrix[0, -1] += layer.twist_elongation_entry(radius_1, radius_2)

            # Fill the twist entries
            matrix[-1, i] = layer.first_twist_entry(radius_1, radius_2)
            matrix[-1, i + self.no_of_layers] = layer.second_twist_entry(
                radius_1, radius_2
            )
            matrix[-1, -2] += layer.elongation_twist_entry(radius_1, radius_2)
            matrix[-1, -1] += layer.twist_twist_entry(radius_1, radius_2)
        return matrix

    def create_loading_vector(self, loading: VesselLoad) -> np.ndarray:
        vector = np.zeros((self.no_of_equations, 1))
        vector[0, 0] = -loading.inner_pressure
        vector[-3, 0] = -loading.outer_pressure
        vector[-2, 0] = (
            self.inner_radius**2 * (loading.pressure_difference) / 2
        )

        return vector

    def create_force_derivative_matrix(self):
        matrix = np.zeros((self.no_of_equations, self.no_of_equations))
        print("Force in linear")

        matrix[-2, 0] = self.inner_radius ** (self.first_layer.beta - 1)
        matrix[-2, self.no_of_layers] = self.inner_radius ** (
            -self.first_layer.beta - 1
        )
        matrix[-2, -2] = self.first_layer.alpha_1
        matrix[-2, -1] = self.first_layer.alpha_2 * self.inner_radius

        return matrix


def main():
    pass


if __name__ == "__main__":
    main()


# End
