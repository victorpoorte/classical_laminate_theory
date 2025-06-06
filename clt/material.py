from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass
class LaminaFailureStresses:
    X_t: float
    X_c: float
    Y_t: float
    Y_c: float
    S: float


@dataclass
class Lamina:
    E1: float
    E2: float
    v12: float
    G12: float
    E3: float = None
    G13: float = None
    v13: float = None
    G23: float = None
    v23: float = None
    alpha1: float = None
    alpha2: float = None
    density: float = None
    failure_stresses: LaminaFailureStresses = None
    degraded_matrix: bool = False
    degraded_fibre: bool = False
    p12_negative: float = None
    p12_positive: float = None
    name: str = None

    def __post_init__(self):
        if self.E3 is None:
            self.E3 = self.E2
        if self.G13 is None:
            self.G13 = self.G12

    @cached_property
    def v21(self):
        return self.E2 * self.v12 / self.E1

    @cached_property
    def Q_xx(self):
        return self.E1 / (1 - self.v12 * self.v21)

    @cached_property
    def Q_yy(self):
        return self.E2 / (1 - self.v12 * self.v21)

    @cached_property
    def Q_xy(self):
        return self.v12 * self.E2 / (1 - self.v12 * self.v21)

    @cached_property
    def Q_ss(self):
        return self.G12

    @cached_property
    def Q_matrix(self):
        return np.array(
            [
                [self.Q_xx, self.Q_xy, 0],
                [self.Q_xy, self.Q_yy, 0],
                [0, 0, self.Q_ss],
            ]
        )

    @cached_property
    def Q_vector(self):
        return np.array([[self.Q_xx], [self.Q_yy], [self.Q_xy], [self.Q_ss]])

    @cached_property
    def S11(self):
        return 1 / self.E1

    @cached_property
    def S12(self):
        return -self.v12 / self.E1

    @cached_property
    def S13(self):
        return -self.v12 / self.E1

    @cached_property
    def S22(self):
        return 1 / self.E2

    @cached_property
    def S33(self):
        return 1 / self.E3

    @cached_property
    def S23(self):
        return -self.v23 / self.E2

    @cached_property
    def S44(self):
        return 1 / self.G23

    @cached_property
    def S55(self):
        return 1 / self.G13

    @cached_property
    def S66(self):
        return 1 / self.G12

    @cached_property
    def stiffness_matrix(self):
        return np.array(
            [
                [self.S11, self.S12, self.S13, 0, 0, 0],
                [self.S12, self.S22, self.S23, 0, 0, 0],
                [self.S13, self.S23, self.S33, 0, 0, 0],
                [0, 0, 0, self.S44, 0, 0],
                [0, 0, 0, 0, self.S55, 0],
                [0, 0, 0, 0, 0, self.S66],
            ]
        )

    @cached_property
    def compliance_matrix(self):
        try:
            return np.linalg.inv(self.stiffness_matrix)
        except np.linalg.LinAlgError:
            return None

    @property
    def thermal_expansion_vector(self) -> np.ndarray:
        return np.array([[self.alpha1], [self.alpha2], [0]])

    def _get_compliance_matrix(self, strain_vector: np.ndarray) -> np.ndarray:
        shape = strain_vector.shape
        if shape == (3,):
            return self.Q_matrix
        if shape == (6,):
            return self.compliance_matrix
        raise ValueError("Invalid strain vector shape...")

    def compute_local_stress(
        self, strain_vector: np.ndarray, temperature_delta: float
    ) -> np.ndarray:
        actual_strain = self.compute_actual_strain(
            strain_vector, temperature_delta
        )
        compliance = self._get_compliance_matrix(actual_strain)
        return np.dot(compliance, actual_strain)

    @staticmethod
    def reshape_strain_vector(strain_vector: np.ndarray) -> np.ndarray:
        if strain_vector.shape == (6, 1):
            strain_vector = np.array(
                [
                    strain_vector[0, 0],
                    strain_vector[1, 0],
                    strain_vector[-1, 0],
                ]
            )
        return strain_vector

    def get_alpha_vector(self, strain_vector: np.ndarray) -> np.ndarray:
        if strain_vector.shape == (3,):
            return np.array([self.alpha1, self.alpha2, 0])
        if strain_vector.shape == (6,):
            return np.array([self.alpha1, self.alpha2, self.alpha2, 0, 0, 0])
        raise ValueError("Invalid strain vector shape")

    def compute_actual_strain(
        self, strain_vector: np.ndarray, temperature_delta: float
    ) -> np.ndarray:
        # Remove the free temperature strains from the total strain
        if temperature_delta == 0:
            return strain_vector
        return (
            strain_vector
            - self.get_alpha_vector(strain_vector) * temperature_delta
        )

    def create_inter_lamina_material(self) -> Lamina:

        raise NotImplementedError()

        from copy import deepcopy

        my_copy = deepcopy(self)
        attrs = [
            ("E1", "E2"),
            ("v12", "v23"),
            ("G12", "G23"),
            ("alpha1", "alpha2"),
        ]
        for original, target in attrs:
            setattr(my_copy, original, getattr(my_copy, target))
        if my_copy.failure_stresses is not None:
            my_copy.failure_stresses.longitudinal = (
                my_copy.failure_stresses.traverse
            )
        return my_copy

    def get_failure_stress(self, temperature: float) -> LaminaFailureStresses:
        self.failure_stresses: dict

        if temperature in self.failure_stresses:
            return self.failure_stresses[temperature]
        
        str_key = str(temperature)
        if str_key in self.failure_stresses:
            return self.failure_stresses[str_key]
        
        raise KeyError(f"Temperature {temperature} not found in failure_stresses.")

    @classmethod
    def T300_934(cls):
        """Create an instance of the T300/934 material, as used in the papers of Xia"""
        return cls(
            E1=141.6e9,
            E2=10.7e9,
            v12=0.268,
            G12=3.88e9,
            v23=0.495,
            G23=3.88e9,
            p12_negative = 0.3,
            p12_positive = 0.3,
            alpha1=0.006e-6,
            alpha2=30.04e-6,
            density=1535,
        )

    @classmethod
    def T300_N5208(cls):
        """Create an instance of the T300/N5208 material, as used in the papers of Parnas"""
        X_t = 1500e6
        X_c = -1500e6
        Y_t = 40e6
        Y_c = -146e6
        S = 68e6
        failure_stresses = LaminaFailureStresses(X_t, X_c, Y_t, Y_c, S)

        # Material definition
        return cls(
            E1=181e9,
            E2=10.3e9,
            v12=0.28,
            G12=7.17e9,
            v23=0.28,
            G23=7.17e9,
            p12_negative = 0.3,
            p12_positive = 0.3,
            alpha1=0.02e-6,
            alpha2=22.5e-6,
            density=1600,
            failure_stresses={"293": failure_stresses},
            name="T300/N5208",
        )


class MaterialFactory:
    _materials = {
        "T300/934": Lamina.T300_934(),
        "T300/N5208": Lamina.T300_N5208(),
    }

    @property
    def available(self):
        return ", ".join(self._materials.keys())

    def create_material(self, material: str) -> Lamina:
        mat = self._materials.get(material)
        if mat is None:
            raise ValueError(
                f"\n{material} not available...\n"
                + f"Available materials are: {self.available}.\n" 
            )
        return mat

def main():
    pass


if __name__ == "__main__":
    main()


# End
