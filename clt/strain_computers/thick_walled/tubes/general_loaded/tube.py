from abc import abstractmethod
from typing import Protocol
import numpy as np


from ..tube import Tube
from ..tube_layers import TubeLayer


class VesselLoad(Protocol):
    inner_pressure: float
    outer_pressure: float
    pressure_difference: float
    temperature_delta: float
    axial_load: float


class TubeLayer(TubeLayer):
    beta: float
    alpha_1: float
    alpha_2: float
    load_factor: float
    load_vector_elongation_factor: float
    load_vector_twist_factor: float

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

    @abstractmethod
    def load_vector_stress_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def load_vector_displacement_entry(self, radius: float) -> float:
        ...

    @abstractmethod
    def load_vector_elongation_entry(
        self, radius1: float, radius2: float
    ) -> float:
        ...

    @abstractmethod
    def load_vector_twist_entry(self, radius1: float, radius2: float) -> float:
        ...

    @abstractmethod
    def d_epsilon_theta_d_solution(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> np.ndarray:
        ...

    @abstractmethod
    def load_derivative_elongation_entry(
        self,
        radius1: float,
        radius2: float,
        epsilon_theta: float,
        d_epsilon_d_solution: np.ndarray,
    ) -> np.ndarray:
        ...

    @abstractmethod
    def load_derivative_twist_entry(
        self,
        radius1: float,
        radius2: float,
        epsilon_theta: float,
        d_epsilon_d_solution: np.ndarray,
    ) -> np.ndarray:
        ...

    @abstractmethod
    def load_derivative_displacement_entry(
        self, d_epsilon_d_solution: np.ndarray
    ) -> float:
        ...


class Tube(Tube):
    layers: list[TubeLayer]
    first_layer: TubeLayer
    last_layer: TubeLayer

    @staticmethod
    def determine_number_of_equations(number_of_layers: int) -> int:
        return number_of_layers * 2 + 2

    @property
    def no_of_equations(self):
        return self.determine_number_of_equations(self.no_of_layers)

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
        vector[-3, 0] = (
            -load.outer_pressure
            - self.last_layer.load_vector_stress_entry(self.outer_radius)
            * load.temperature_delta
        )

        # Extension of the vessel
        vector[-2, 0] += self.inner_radius**2 * load.pressure_difference / 2
        vector[-2, 0] += load.axial_load / (2 * np.pi)
        for i, layer in enumerate(self.layers):
            vector[-2, 0] -= (
                layer.load_vector_elongation_entry(
                    self.radii[i], self.radii[i + 1]
                )
                * load.temperature_delta
                / 2
            )

        for i, layer in enumerate(self.layers):
            vector[-1, 0] += (
                layer.load_vector_twist_entry(self.radii[i], self.radii[i + 1])
                * load.temperature_delta
            )

        return vector

    def create_force_derivative_matrix(
        self,
        original_radii: list[float],
        loading: VesselLoad,
        solution_vector: np.ndarray,
    ) -> np.ndarray:
        matrix = np.zeros((self.no_of_equations, self.no_of_equations))

        # Add entries associated with thermal loading
        for layer_index, layer in enumerate(self.layers[:-1]):
            # Extract layer and radius for boundary condition computations
            inner_radius = original_radii[layer_index]
            outer_radius = original_radii[layer_index + 1]
            interface_radius = self.radii[layer_index + 1]
            layer2 = self.layers[layer_index + 1]

            # Compute theta strain for current layer and solution vector
            global_strain = layer.compute_global_strain_vector(
                interface_radius,
                layer_index,
                solution_vector,
                loading.temperature_delta,
            )
            epsilon_theta = global_strain[1]

            # Derivative vector of theta strain wrt solution vector
            d_epsilon_d_solution = layer.d_epsilon_theta_d_solution(
                interface_radius,
                solution_vector,
                layer_index,
                self.no_of_layers,
            )

            # Add temperature associated displacement derivative
            matrix[layer_index + 1] = (
                layer2.load_derivative_displacement_entry(
                    interface_radius, epsilon_theta, d_epsilon_d_solution
                )
                - layer.load_derivative_displacement_entry(
                    interface_radius, epsilon_theta, d_epsilon_d_solution
                )
            ) * loading.temperature_delta

            matrix[-2] += (
                layer.load_derivative_elongation_entry(
                    outer_radius,
                    epsilon_theta,
                    d_epsilon_d_solution,
                )
                - layer.load_derivative_elongation_entry(
                    inner_radius,
                    epsilon_theta,
                    d_epsilon_d_solution,
                )
            ) * loading.temperature_delta

            matrix[-1] += (
                layer.load_derivative_twist_entry(
                    outer_radius,
                    epsilon_theta,
                    d_epsilon_d_solution,
                )
                - layer.load_derivative_twist_entry(
                    inner_radius,
                    epsilon_theta,
                    d_epsilon_d_solution,
                )
            ) * loading.temperature_delta

        # Add effects associated to the change in axial loading associated with
        # the change in radius, loading to different pressure surface
        original_inner_radius = original_radii[0]
        epsilon_theta = self.first_layer.compute_global_strain_vector(
            original_inner_radius,
            0,
            solution_vector,
            loading.temperature_delta,
        )[1]
        matrix[-2] += (
            original_inner_radius**2
            * loading.pressure_difference
            * (1 + epsilon_theta)
            * self.first_layer.d_epsilon_theta_d_solution(
                interface_radius,
                solution_vector,
                0,
                self.no_of_layers,
            )
        )

        return matrix


def main():
    pass


if __name__ == "__main__":
    main()


# End
