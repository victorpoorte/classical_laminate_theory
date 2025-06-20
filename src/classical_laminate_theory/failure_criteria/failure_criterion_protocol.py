from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from abc import abstractmethod
from typing import Protocol

import numpy as np

from ..layer import layer_to_laminate_layer


class FailureStresses(Protocol):
    X_t: float
    X_c: float
    Y_t: float
    Y_c: float
    S: float


class Lamina:
    p12_negative: float 
    p12_positive: float 
    def get_failure_stress(self, operating_temperature: float) -> FailureStresses:
        ...


class Layer(Protocol):
    lamina: Lamina
    @abstractmethod
    def compute_local_stress(
        self, local_strain: np.ndarray, temperature_delta: float
    ) -> np.ndarray:
        ...

    @abstractmethod
    def compute_local_strain(self, global_strain: np.ndarray) -> np.ndarray:
        ...


class FailureType(Enum):
    LONGITUDINAL_TENSION = "longitudinal_tension"
    LONGITUDINAL_COMPRESSION = "longitudinal_compression"
    TRANSVERSE_TENSION = "transverse_tension"
    TRANSVERSE_COMPRESSION = "transverse_compression"

    @classmethod
    def types(cls):
        return [mode for mode in cls.__iter__()]

    @classmethod
    def names(cls):
        return [mode.value for mode in cls.types()]

    @classmethod
    def short_labels(cls):
        return ["".join([name[0].upper() for name in mode.split("_")]) for mode in cls.names()]


@dataclass
class FailureMode:
    criterion: str
    type: FailureType
    angle: float | None = None # Optional only if there is a failure angle
    

def classify_failure_mode(long_factor: float, long_tension: bool, tran_factor: float, tran_tension: bool) -> FailureType:
    if long_factor <= tran_factor:
        if long_tension:
            return FailureType.LONGITUDINAL_TENSION
        return FailureType.LONGITUDINAL_COMPRESSION
    if tran_tension:
        return FailureType.TRANSVERSE_TENSION
    return FailureType.TRANSVERSE_COMPRESSION
    

class FailureCriterion(Protocol):
    """Failure Criterion protocol class. This is to be used as a reference
    for the definition of other failure criteria.

    The method to be defined is the 'determine_failure_indices' method.
    This is to return a tuple of failure indices for each direction.
    """
    name: str

    def __str__(self):
        return self.name.capitalize()

    @abstractmethod
    def factor_of_safety(self, global_strain: np.ndarray, layer: Layer, temperature_delta: float, operating_temperature: float) -> tuple[float, FailureMode]:
        ...

    @abstractmethod
    def long_and_trav_index(self, global_strain: np.ndarray, layer: Layer, temperature_delta: float, operating_temperature: float) -> tuple[float, float]:
        ...
        
    @staticmethod
    def _compute_layer_stress_state(
        layer: Layer, global_strain: np.ndarray, temperature_delta: float
    ) -> StressState:
        laminate_layer = layer_to_laminate_layer(layer)
        return StressState(
            laminate_layer.compute_local_stress(
                laminate_layer.compute_local_strain(global_strain), temperature_delta
            )
        )
    
    @staticmethod
    def _tension(stress: float) -> bool:
        """Method to determine if the stress is tensile. Tension is defined
        as positive, and compression negative.

        Args:
            stress (float): Stress value

        Returns:
            bool: Bool indicating tension.
        """
        return stress >= 0
    

class StressState:
    def __init__(self, stress_vector: np.ndarray) -> None:
        shape = stress_vector.shape
        if shape == (6,):
            self.initialise_3D(stress_vector)
            return None
        if shape == (3,):
            self.initialise_2D(stress_vector)
            return None
        raise ValueError(f"Invalid stress vector shape: {shape}")

    def __eq__(self, other: StressState) -> bool:
        for stress1, stress2 in zip(self.vector, other.vector):
            if stress1 != stress2:
                return False
        return True

    def initialise_3D(self, stress_vector: np.ndarray) -> None:
        self.vector = stress_vector[:,np.newaxis]

    def initialise_2D(self, stress_vector: np.ndarray) -> None:
        self.vector = np.zeros((6, 1))
        self.vector[0, 0] = stress_vector[0]
        self.vector[1, 0] = stress_vector[1]
        self.vector[5, 0] = stress_vector[2]

    @property
    def longitudinal(self):
        return self.vector[0, 0]

    @property
    def traverse(self):
        return self.vector[1, 0]

    @property
    def shear(self):
        return self.vector[5, 0]


def main():
    pass


if __name__ == "__main__":
    main()


# End
