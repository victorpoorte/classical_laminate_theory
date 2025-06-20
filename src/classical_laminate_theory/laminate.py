from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import numpy as np

from .laminate_layer import LaminateLayer


class Orientation(Protocol):
    rotation: float

    @abstractmethod
    def compute_local_strains(self, strain: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def is_90_layer(self) -> bool:
        ...


class LaminateLoad(Protocol):
    vector: np.ndarray
    temperature_delta: float


@dataclass
class Laminate:
    layers: list[LaminateLayer]

    @property
    def number_of_layers(self):
        return len(self.layers)

    @cached_property
    def layer_thicknesses(self):
        return np.array([layer.thickness for layer in self.layers])

    @cached_property
    def thickness(self) -> float:
        """Method to determine the thickness of the laminate."""
        return sum(self.layer_thicknesses)

    @cached_property
    def z_locations(self) -> list[LaminateLayer]:
        """Determine the z locations of the layers."""
        self.z_locations = np.zeros(self.number_of_layers + 1)
        self.z_locations[0] = -0.5 * self.thickness
        for i, layer in enumerate(self.layers, 1):
            self.z_locations[i] = self.z_locations[i - 1] + layer.thickness

        return self.z_locations

    @cached_property
    def A(self) -> np.ndarray:
        """Method to determine the A matrix of the laminate."""
        return sum(
            [
                layer.Q_matrix
                * (self.z_locations[k] - self.z_locations[k - 1])
                for k, layer in enumerate(self.layers, 1)
            ]
        )

    @cached_property
    def B(self) -> np.ndarray:
        """Method to determine the B matrix of the laminate."""
        return sum(
            [
                0.5
                * layer.Q_matrix
                * (self.z_locations[k] ** 2 - self.z_locations[k - 1] ** 2)
                for k, layer in enumerate(self.layers, 1)
            ]
        )

    @cached_property
    def D(self) -> np.ndarray:
        """Method to determine the C matrix of the laminate."""
        return sum(
            [
                1
                / 3
                * layer.Q_matrix
                * (self.z_locations[k] ** 3 - self.z_locations[k - 1] ** 3)
                for k, layer in enumerate(self.layers, 1)
            ]
        )

    @cached_property
    def Ex(self) -> float:
        """Method to determine the  axial stiffness."""
        return (self.A[0, 0] * self.A[1, 1] - self.A[0, 1] ** 2) / (
            self.thickness * self.A[1, 1]
        )

    @cached_property
    def Ey(self) -> float:
        """Function to determine the traverse stiffness."""
        return (self.A[0, 0] * self.A[1, 1] - self.A[0, 1] ** 2) / (
            self.thickness * self.A[0, 0]
        )

    @cached_property
    def Gxy(self) -> float:
        """Function to determine the bending stiffness"""
        return self.A[2, 2] / self.thickness

    @cached_property
    def v_xy(self) -> float:
        """Method to determine the Poisson ration in xy direction"""
        return self.A[0, 1] / self.A[1, 1]

    @cached_property
    def v_yx(self) -> float:
        """Method to determine the Poisson ration in xy direction"""
        return self.A[0, 1] / self.A[0, 0]

    @cached_property
    def stiffness_matrix(self) -> np.ndarray:
        top_row = np.concatenate((self.A, self.B), axis=1)
        bottom_row = np.concatenate((self.B, self.D), axis=1)
        self.stiffness_matrix = np.concatenate((top_row, bottom_row), axis=0)

        return self.stiffness_matrix

    @cached_property
    def compliance_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.stiffness_matrix)

    @cached_property
    def z_mid_locations(self):
        return np.array(
            [
                (self.z_locations[i] + self.z_locations[i + 1]) / 2
                for i, _ in enumerate(self.z_locations[:-1])
            ]
        )

    @cached_property
    def thermal_extension_vector(self) -> np.ndarray:
        load_vector = sum(
            [layer.thermal_extension_load_entry for layer in self.layers]
        )
        return np.dot(np.linalg.inv(self.A), load_vector)

    @cached_property
    def thermal_moment_vector(self) -> np.ndarray:
        return sum(
            [
                0.5
                * np.dot(layer.Q_matrix, layer.thin_thermal_expansion_vector)
                * (self.z_locations[k] ** 2 - self.z_locations[k - 1] ** 2)
                for k, layer in enumerate(self.layers, 1)
            ]
        )

    @cached_property
    def thermal_load_vector(self) -> np.ndarray:
        return np.vstack(
            (self.thermal_extension_vector, self.thermal_moment_vector)
        )

    def total_load_vector(self, laminate_load: LaminateLoad) -> np.ndarray:
        if laminate_load.temperature_delta == 0:
            return laminate_load.vector
        return (
            laminate_load.vector
            + laminate_load.temperature_delta * self.thermal_load_vector
        )

    def compute_global_strain(self, laminate_load: LaminateLoad) -> np.ndarray:
        return np.dot(
            self.compliance_matrix, self.total_load_vector(laminate_load)
        )

    def compute_total_strains(self, global_strains: np.ndarray) -> np.ndarray:
        return global_strains[:3] + global_strains[3:] * self.z_mid_locations

    @property
    def C_bar(self) -> np.ndarray:
        return (
            sum(layer.thickness * layer.C_bar for layer in self.layers)
            / self.thickness
        )

    @property
    def C11(self) -> float:
        return self.C_bar[0, 0]

    @property
    def C22(self) -> float:
        return self.C_bar[1, 1]

    @property
    def C33(self) -> float:
        return self.C_bar[2, 2]

    @property
    def C12(self) -> float:
        return self.C_bar[0, 1]

    @property
    def C13(self) -> float:
        return self.C_bar[0, 2]

    @property
    def C23(self) -> float:
        return self.C_bar[1, 2]

    @property
    def C16(self) -> float:
        return self.C_bar[0, 5]

    @property
    def C26(self) -> float:
        return self.C_bar[1, 5]

    @property
    def C36(self) -> float:
        return self.C_bar[2, 5]

    @property
    def C66(self) -> float:
        return self.C_bar[5, 5]

    @property
    def out_of_plane_thermal_extension(self) -> float:
        return (
            sum(layer.thickness * layer.lamina.alpha2 for layer in self.layers)
            / self.thickness
        )

    @property
    def thick_thermal_expansion_vector(self):
        return np.array(
            [
                [self.thermal_extension_vector[0, 0]],
                [self.thermal_extension_vector[1, 0]],
                [self.out_of_plane_thermal_extension],
                [0],
                [0],
                [self.thermal_extension_vector[-1, 0]],
            ]
        )


def main():
    pass


if __name__ == "__main__":
    main()


# End
