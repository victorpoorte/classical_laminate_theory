from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .loading import convert_load_array_to_laminate_load

from .laminate_factory import LaminateFactoryProtocol, TraditionalLaminateFactory


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
    ):
        ...

class Laminate(Protocol):

    def compute_global_strain(self, load: Load) -> np.ndarray:
        ...

    def compute_total_strains(self, global_strain: np.ndarray) -> np.ndarray:
        ...



@dataclass
class ClassicalLaminateTheoryComputer(StrainComputer):
    laminate_factory: LaminateFactoryProtocol = TraditionalLaminateFactory()

    def compute_global_strains(
        self,
        layers: list[Layer],
        load: Load,
    ) -> np.ndarray:
        laminate = self.laminate_factory.create_laminate(layers)
        global_strain = laminate.compute_global_strain(load)
        return laminate.compute_total_strains(global_strain)
    

class StrainComputerFactory():
    _strain_computers = {
        "classical_laminate": ClassicalLaminateTheoryComputer()
    }

    @property
    def available(self):
        return ", ".join(self._strain_computers.keys())

    def create(self, computer_type: str) -> StrainComputer:
        computer = self._strain_computers.get(computer_type)
        if computer is None:
            raise ValueError(
                "\nStrain computer not implemented.\n"
                + f"Available computers are: {self.available}"
            )
        return computer



def main():
    pass


if __name__ == "__main__":
    main()


# End
