from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol
import numpy as np


class VesselLoading(Protocol):
    ...


class TubeLayer(Protocol):
    thickness: float


class LaminateLayer(Protocol):
    ...


@dataclass
class Tube:
    inner_radius: float
    layers: list[TubeLayer]

    @property
    def no_of_layers(self):
        return len(self.layers)

    @staticmethod
    @abstractmethod
    def determine_number_of_equations(self, number_of_layers: int) -> int:
        ...

    @property
    @abstractmethod
    def no_of_equations(self):
        ...

    @property
    def thicknesses(self):
        return [layer.thickness for layer in self.layers]

    @property
    def thickness(self):
        return sum(self.thicknesses)

    @property
    def outer_radius(self):
        return self.inner_radius + self.thickness

    @property
    def radii(self):
        radii = [self.inner_radius] * (self.no_of_layers + 1)
        for i, layer in enumerate(self.layers):
            radii[i + 1] = radii[i] + layer.thickness

        return radii

    @property
    def mid_layer_radii(self):
        return [
            (radius + self.radii[i]) / 2
            for i, radius in enumerate(self.radii[1:])
        ]

    @property
    def first_layer(self) -> TubeLayer:
        return self.layers[0]

    @property
    def last_layer(self) -> TubeLayer:
        return self.layers[-1]

    # def get_layers(
    #     self, radius: float
    # ) -> tuple[int, TubeLayer, LaminateLayer]:
    #     for i, tube_layer in enumerate(self.layers):
    #         if self.radii[i] <= radius and radius <= self.radii[i + 1]:
    #             return i, tube_layer, self.laminate_layers[i]
    #     if radius <= self.outer_radius:
    #         return i, tube_layer, self.laminate_layers[i]
    #     raise ValueError("Provided radius out of bound for tube")
    
    def get_layers(
        self, radius: float
    ) -> tuple[int, TubeLayer, LaminateLayer]:
        for i, tube_layer in enumerate(self.layers):
            if self.radii[i] <= radius and radius <= self.radii[i + 1]:
                return i, tube_layer
        if radius <= self.outer_radius:
            return i, tube_layer
        raise ValueError("Provided radius out of bound for tube")

    def create_matrix(self):
        matrix = np.zeros((self.no_of_equations, self.no_of_equations))

        matrix[
            : self.no_of_layers * 2
        ] = self.through_thickness_compatibility()

        matrix[self.no_of_layers * 2 :] = self.longitudinal_compatibility()

        return matrix

    @abstractmethod
    def create_loading_vector(
        self, inner_pressure: float, outer_pressure: float, _: float
    ) -> np.ndarray:
        ...

    def through_thickness_compatibility(self) -> np.ndarray:
        matrix = np.zeros((self.no_of_layers * 2, self.no_of_equations))
        matrix[0, :] = self.internal_stress_row()

        for i in range(self.no_of_layers - 1):
            matrix[i + 1] = self.displacement_row(
                self.layers[i], self.layers[i + 1], i
            )
            matrix[i + self.no_of_layers] = self.stress_row(
                self.layers[i], self.layers[i + 1], i
            )

        matrix[-1] = self.external_stress_row()

        return matrix

    @abstractmethod
    def internal_stress_row(self):
        ...

    @abstractmethod
    def displacement_row(
        self, layer1: TubeLayer, layer2: TubeLayer
    ) -> np.ndarray:
        ...

    @abstractmethod
    def stress_row(self, layer1: TubeLayer, layer2: TubeLayer) -> np.ndarray:
        ...

    @abstractmethod
    def external_stress_row(self):
        ...

    @abstractmethod
    def longitudinal_compatibility(self) -> np.ndarray:
        ...

    @abstractmethod
    def create_force_derivative_matrix(self) -> np.ndarray:
        ...

    def compute_solution_vector(self, load: VesselLoading) -> np.ndarray:
        return np.dot(
            np.linalg.inv(self.create_matrix()),
            self.create_loading_vector(load),
        )


def main():
    pass


if __name__ == "__main__":
    main()


# End
