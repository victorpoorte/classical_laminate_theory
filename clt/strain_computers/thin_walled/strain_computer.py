from dataclasses import dataclass
import numpy as np

from common import Load
from classical_laminate_theory import ClassicalLaminateTheoryComputer, TraditionalLaminateFactory

from ..computer_protocol import PressureVesselStrainComputer, Layer

@dataclass
class LaminateLoad(Load):
    radius: float

    @property
    def pressure_difference(self):
        return self.inner_pressure - self.outer_pressure

    @property
    def Nx(self) -> float:
        return self.pressure_difference * self.radius / 2

    @property
    def Ny(self) -> float:
        return self.pressure_difference * self.radius

    @property
    def Nxy(self):
        return 0

    @property
    def Mx(self):
        return 0

    @property
    def My(self):
        return 0

    @property
    def Mxy(self):
        return 0

    @property
    def vector(self):
        return np.array(
            [
                [self.Nx],
                [self.Ny],
                [self.Nxy],
                [self.Mx],
                [self.My],
                [self.Mxy],
            ]
        )
    
def load_to_laminate_load(load: Load, radius: float) -> LaminateLoad:
    return LaminateLoad(**({key: getattr(load, key) for key in load.attributes} | {"radius": radius}))


class ThinWalledVesselStrainComputer(PressureVesselStrainComputer):

    def __init__(self, laminate_strain_computer_initialiser: ClassicalLaminateTheoryComputer = ClassicalLaminateTheoryComputer, laminate_factory: TraditionalLaminateFactory = TraditionalLaminateFactory()) -> None:
        self._strain_computer: ClassicalLaminateTheoryComputer = laminate_strain_computer_initialiser(laminate_factory)

    def __str__(self) -> str:
        return "Linear Thin-Walled"

    def compute_global_strains(self, layers: list[Layer], load: Load, radius: float) -> np.ndarray:
        laminate_load = load_to_laminate_load(load, radius)
        return self._strain_computer.compute_global_strains(layers, laminate_load)
    
# End
    