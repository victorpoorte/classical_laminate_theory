import math
from dataclasses import dataclass
from functools import cached_property

import warnings

import numpy as np


WARNING_THROWN = False
CHECK_ROTATION =  False

@dataclass
class Orientation:
    rotation: float
    degree: bool = True

    def __str__(self) -> str:
        return f"layer of {self.rotation_degree:.1f}"

    def __post_init__(self):
        if CHECK_ROTATION:
            self.check_rotation()

        # Convert the orientation  into  radians if required
        if self.degree:
            self.rotation = np.deg2rad(self.rotation)

    def check_rotation(self) -> bool:
        global WARNING_THROWN
        if (
            self.rotation > 0
            and self.rotation < np.pi / 2
            and self.degree
            and not WARNING_THROWN
        ):
            WARNING_THROWN = True
            warnings.warn(
                f"\nValue of rotation ({self.rotation}) seems invalid."
                +"\nMake sure that degrees and radians are used in the right way\n"
            )
        return True

    @property
    def rotation_degree(self):
        return math.degrees(self.rotation)

    @cached_property
    def m(self):
        return np.cos(self.rotation)

    @cached_property
    def n(self):
        return np.sin(self.rotation)

    @cached_property
    def rotation_matrix(self) -> None:
        """Returns the rotation matrix for the given orientation."""
        return np.array(
            [
                [
                    self.m**4,
                    self.n**4,
                    2 * self.m**2 * self.n**2,
                    4 * self.m**2 * self.n**2,
                ],
                [
                    self.n**4,
                    self.m**4,
                    2 * self.m**2 * self.n**2,
                    4 * self.m**2 * self.n**2,
                ],
                [
                    self.m**2 * self.n**2,
                    self.m**2 * self.n**2,
                    -2 * self.m**2 * self.n**2,
                    (self.m**2 - self.n**2) ** 2,
                ],
                [
                    self.m**2 * self.n**2,
                    self.m**2 * self.n**2,
                    self.m**4 + self.n**4,
                    -4 * self.m**2 * self.n**2,
                ],
                [
                    self.m**3 * self.n,
                    -self.m * self.n**3,
                    self.m * self.n**3 - self.m**3 * self.n,
                    2 * (self.m * self.n**3 - self.m**3 * self.n),
                ],
                [
                    self.m * self.n**3,
                    -self.m**3 * self.n,
                    self.m**3 * self.n - self.m * self.n**3,
                    2 * (self.m**3 * self.n - self.m * self.n**3),
                ],
            ]
        )

    @cached_property
    def strain_rotation_matrix_2D(self) -> np.ndarray:
        return np.array(
            [
                [self.m**2, self.n**2, self.m * self.n],
                [self.n**2, self.m**2, -self.m * self.n],
                [
                    -2 * self.m * self.n,
                    2 * self.m * self.n,
                    self.m**2 - self.n**2,
                ],
            ]
        )

    @cached_property
    def strain_rotation_matrix_3D(self) -> np.ndarray:
        return np.array(
            [
                [self.m**2, self.n**2, 0, 0, 0, self.m * self.n],
                [self.n**2, self.m**2, 0, 0, 0, -self.m * self.n],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, self.m, -self.n, 0],
                [0, 0, 0, self.n, self.m, 0],
                [
                    -2 * self.m * self.n,
                    2 * self.m * self.n,
                    0,
                    0,
                    0,
                    self.m**2 - self.n**2,
                ],
            ]
        )

    @cached_property
    def stress_rotation_matrix_3D(self):
        return np.array(
            [
                [self.m**2, self.n**2, 0, 0, 0, 2 * self.m * self.n],
                [self.n**2, self.m**2, 0, 0, 0, -2 * self.m * self.n],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, self.m, -self.n, 0],
                [0, 0, 0, self.n, self.m, 0],
                [
                    -self.m * self.n,
                    self.m * self.n,
                    0,
                    0,
                    0,
                    self.m**2 - self.n**2,
                ],
            ]
        )

    @cached_property
    def T_2_rotation_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.m**2, self.n**2, -self.m * self.n],
                [self.n**2, self.m**2, self.m * self.n],
                [
                    2 * self.m * self.n,
                    -2 * self.m * self.n,
                    self.m**2 - self.n**2,
                ],
            ]
        )

    def compute_local_strains(self, strain_vector: np.ndarray) -> np.ndarray:
        return np.dot(self._get_rotation_matrix(strain_vector), strain_vector)

    def _get_rotation_matrix(self, strain_vector: np.ndarray) -> np.ndarray:
        shape = strain_vector.shape
        if shape == (3,):
            return self.strain_rotation_matrix_2D
        if shape == (6,):
            return self.strain_rotation_matrix_3D
        raise ValueError(f"Invalid strain vector shape: {shape}")

    @property
    def is_90_degrees(self) -> bool:
        return abs(self.rotation - np.pi / 2) == 0


def main():
    pass


if __name__ == "__main__":
    main()


# End
