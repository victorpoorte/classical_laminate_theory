from dataclasses import dataclass
from typing import Protocol
import numpy as np


from ..loading import LaminateLoad
from .failure_analyser import FailureAnalyser
from .failure_strategy import FailureResult

class Layer(Protocol):
    ...

class FailureEnvelopeGenerator:
    def __init__(self, analyser: FailureAnalyser, x_axis: str, y_axis: str, angle_resolution: float = 1.0) -> None:
        self.analyser = analyser
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.angle_resolution = angle_resolution

        self._create_angles()
        self._create_loads()

    def compute_envelope(self, layers: list[Layer]) -> list[FailureResult]:
        envelope_loads = [
            self.analyser.analyse(layers, load)
            for load in self.loads
        ]

        return FailureEnvelope(envelope_loads, self.angles)

    def _create_angles(self):
        self.angles = np.radians(np.arange(0, 360+self.angle_resolution, self.angle_resolution))
        return self.angles
    
    def _create_loads(self) -> list[LaminateLoad]:
        self.loads = [
            LaminateLoad._create_angle_based_load(self.x_axis, self.y_axis, angle)
            for angle in self.angles
        ]

        return self.loads
    
@dataclass
class FailureEnvelope:
    results: list[FailureResult]
    angles: np.ndarray

    @property
    def first_magnitudes(self):
        return [result.first_load.magnitude for result in self.results]
    
    @property
    def is_last_ply(self):
        return self.results[0].is_last_ply
    
    @property
    def last_magnitudes(self):
        if not self.is_last_ply:
            raise ValueError("Not a last ply failure analysis")
        return [result.last_load.magnitude for result in self.results]   

    @property
    def first_x_values(self):
        return self.first_magnitudes * np.cos(self.angles)
    
    @property
    def first_y_values(self):
        return self.first_magnitudes * np.sin(self.angles)


    @property
    def last_x_values(self):
        return self.last_magnitudes * np.cos(self.angles)
    
    @property
    def last_y_values(self):
        return self.last_magnitudes * np.sin(self.angles)
    


# End
