from abc import abstractmethod
from dataclasses import dataclass
import numpy as np

from typing import Protocol

from common import Load

class Layer(Protocol):
    ...


class PressureVesselStrainComputer(Protocol):

    @abstractmethod
    def compute_global_strains(self, layers: list[Layer], load: Load, radius: float) -> np.ndarray:
        ...

@dataclass
class PressureVesselStrainComputerAdapter:
    strain_computer: PressureVesselStrainComputer
    radius: float

    def compute_global_strains(self, layers: list[Layer], load: Load) -> np.ndarray:
        return self.strain_computer.compute_global_strains(layers, load, self.radius)
    

# End
    