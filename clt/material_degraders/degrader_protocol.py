from abc import abstractmethod
from typing import Protocol

from ..material import Lamina

DEGRADATION_FACTOR = 1e-1
DEGRADED_POISSON = 0.001


class MaterialDegrader(Protocol):
    @abstractmethod
    def degrade_material(self, lamina: Lamina, failure_index: float) -> Lamina:
        ...


def main():
    pass


if __name__ == "__main__":
    main()


# End
