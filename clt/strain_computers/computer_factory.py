from abc import abstractmethod
from typing import Protocol

from classical_laminate_theory.clt.strain_computers import ClassicalLaminateTheoryComputer
from .thick_walled import (
    LinearThickWalledStrainComputer,
    NonLinearThickWalledStrainComputer,
)
from .thin_walled.strain_computer import ThinWalledVesselStrainComputer

class Load(Protocol):
    ...

class Layer(Protocol):
    ...

class StrainComputer(Protocol):
    @abstractmethod
    def compute_global_strains(
        self,
        layers: list[Layer],
        load: Load,
        radius: float,
    ):
        ...


class StrainComputerFactory():
    _strain_computers = {
        "thin_walled_linear": ThinWalledVesselStrainComputer(),
        "thick_walled_linear": LinearThickWalledStrainComputer(),
        "thick_walled_non_linear": NonLinearThickWalledStrainComputer(),
        "classical_laminate": ClassicalLaminateTheoryComputer()
    }

    @property
    def available(self):
        return ", ".join(self._strain_computers.keys())

    def create_strain_computer(self, computer_type: str) -> StrainComputer:
        computer = self._strain_computers.get(computer_type)
        if computer is None:
            raise ValueError(
                "\nStrain computer not implemented.\n"
                + f"Available computers are: {self.available}"
            )
        return computer