from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import numpy as np

from .orientation import Orientation


class Lamina(Protocol):
    Q_vector: np.ndarray
    compliance_matrix: np.ndarray
    thermal_expansion_vector: np.ndarray
    alpha2: float

    @abstractmethod
    def compute_local_stress(self, strain_vector: np.ndarray) -> np.ndarray:
        ...


@dataclass
class LaminateLayer:
    lamina: Lamina
    thickness: float
    rotation: float
    degrees: bool = True

    def __post_init__(self):
        self.orientation = Orientation(self.rotation, degree=self.degrees)

    @cached_property
    def Q_vector(self):
        """Method to determine the ratoated q materix of the layer."""
        self.Q_vector = np.dot(
            self.orientation.rotation_matrix, self.lamina.Q_vector
        )

        return self.Q_vector

    @cached_property
    def Q_matrix(self):
        self.Q_matrix = np.array(
            [
                [
                    self.Q_vector[0, 0],
                    self.Q_vector[3, 0],
                    self.Q_vector[4, 0],
                ],
                [
                    self.Q_vector[3, 0],
                    self.Q_vector[1, 0],
                    self.Q_vector[5, 0],
                ],
                [
                    self.Q_vector[4, 0],
                    self.Q_vector[5, 0],
                    self.Q_vector[2, 0],
                ],
            ]
        )

        return self.Q_matrix

    @cached_property
    def C_bar(self):
        if self.lamina.compliance_matrix is None:
            return None
        try:
            my_inverse = np.linalg.inv(
                self.orientation.stress_rotation_matrix_3D
            )
        except np.linalg.LinAlgError:
            return None
        return np.linalg.multi_dot(
            [
                my_inverse,
                self.lamina.compliance_matrix,
                self.orientation.strain_rotation_matrix_3D,
            ]
        )

    @cached_property
    def C11(self):
        return self.C_bar[0, 0]

    @cached_property
    def C12(self):
        return self.C_bar[0, 1]

    @cached_property
    def C13(self):
        return self.C_bar[0, 2]

    @cached_property
    def C16(self):
        return self.C_bar[0, 5]

    @cached_property
    def C22(self):
        return self.C_bar[1, 1]

    @cached_property
    def C23(self):
        return self.C_bar[1, 2]

    @cached_property
    def C26(self):
        return self.C_bar[1, 5]

    @cached_property
    def C33(self):
        return self.C_bar[2, 2]

    @cached_property
    def C36(self):
        return self.C_bar[2, 5]

    @cached_property
    def C44(self):
        return self.C_bar[3, 3]

    @cached_property
    def C45(self):
        return self.C_bar[3, 4]

    @cached_property
    def C55(self):
        return self.C_bar[4, 4]

    @cached_property
    def C66(self):
        return self.C_bar[5, 5]

    @cached_property
    def is_90_degrees(self) -> bool:
        return self.orientation.is_90_degrees

    @cached_property
    def thin_thermal_expansion_vector(self):
        return np.dot(
            self.orientation.T_2_rotation_matrix,
            self.lamina.thermal_expansion_vector,
        )

    @cached_property
    def thick_thermal_expansion_vector(self):
        return np.array(
            [
                [self.thin_thermal_expansion_vector[0, 0]],
                [self.thin_thermal_expansion_vector[1, 0]],
                [self.lamina.alpha2],
                [0],
                [0],
                [self.thin_thermal_expansion_vector[2, 0]],
            ]
        )

    @cached_property
    def thermal_extension_load_entry(self):
        return self.thickness * np.dot(
            self.Q_matrix, self.thin_thermal_expansion_vector
        )

    def compute_local_strain(self, global_strain: np.ndarray) -> np.ndarray:
        return self.orientation.compute_local_strains(global_strain)

    def compute_local_stress(
        self, local_strain: np.ndarray, temperature_delta: float
    ) -> np.ndarray:
        return self.lamina.compute_local_stress(
            local_strain, temperature_delta
        )


def main():
    pass


if __name__ == "__main__":
    main()


# End
